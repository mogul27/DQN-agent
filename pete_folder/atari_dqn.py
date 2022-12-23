# Try and do the gym's atari breakout

import gym
import numpy as np
import random
import os
from pathlib import Path
import time
import pickle
from collections import deque
import matplotlib.pyplot as plt

import cv2

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim

from dqn_utils import ReplayMemory, DataWithHistory, Timer, Options


class EGreedyPolicy:
    """ Assumes every state has the same possible actions.
    """

    def __init__(self, q_func, options=None):
        """ e-greedy policy based on the supplied q_table.

        :param q_func: Approximates q values for state action pairs so we can select the best action.
        :param options: can contain:
            'epsilon': small epsilon for the e-greedy policy. This is the probability that we'll
                       randomly select an action, rather than picking the best.
            'epsilon_decay_episodes': the number of episodes over which to decay epsilon
            'epsilon_min': the min value epsilon can be after decay
        """
        self.options = Options(options)
        self.options.default('epsilon', 0.1)

        self.q_func = q_func
        self.epsilon = self.options.get('epsilon')
        self.possible_actions = self.q_func.actions
        decay_episodes = self.options.get('epsilon_decay_episodes', 0)
        if decay_episodes == 0:
            self.epsilon_min = self.epsilon
            self.epsilon_decay = 0
        else:
            self.epsilon_min = self.options.get('epsilon_min', 0)
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / decay_episodes

    def select_action(self, state):
        """ The EGreedy policy selects the best action the majority of the time. However, a random action is chosen
        explore_probability amount of times.

        :param state: The sate to pick an action for
        :return: selected action
        """
        # Select an action for the state, use the best action most of the time.
        # However, with probability of explore_probability, select an action at random.
        if np.random.uniform() < self.epsilon:
            action = self.random_action()
        else:
            action = self.q_func.best_action_for(state)

        return action

    def random_action(self):
        return random.choice(self.possible_actions)

    def decay_epsilon(self):
        # decay epsilon for next time - needs to be called at the end of an episode.
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)


# PyTorch models inherit from torch.nn.Module
class QNetwork(nn.Module):
    """ Define neural network to be used by the DQN as both q-network and target-network

    """
    def __init__(self, options):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, len(options.get('actions')))

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.flatten1(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FunctionApprox:

    def __init__(self, options):
        self.options = Options(options)

        self.actions = self.options.get('actions')
        self.q_hat = self.build_neural_network()
        self.q_hat_target = self.build_neural_network()
        # set target weights to match q-network
        self.q_hat_target.load_state_dict(self.q_hat.state_dict())

        self.loss_fn = self.create_loss_function()
        self.optimizer = self.create_optimizer()
        self.discount_factor = self.options.get('discount_factor', 0.99)

        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        if self.work_dir is not None:
            # location for files.
            self.work_dir = Path(self.work_dir)
            if not self.work_dir.exists():
                self.work_dir.mkdir(parents=True, exist_ok=True)

    def get_weights_file(self):
        if self.work_dir is not None:
            weights_file_name = self.options.get('weights_file', f"{self.options.get('env_name')}.pth")
            return self.work_dir / weights_file_name
        return None

    def save_weights(self):
        if self.options.get('save_weights', default=False):
            weights_file = self.get_weights_file()
            if weights_file is not None:
                if not weights_file.parent.exists():
                    weights_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.q_hat.state_dict(), weights_file)

    def load_weights(self):
        if self.options.get('load_weights', default=False):
            weights_file = self.get_weights_file()
            if weights_file is not None and weights_file.exists():
                state_dict = torch.load(weights_file)
                self.set_value_network_weights(state_dict)
                self.set_target_network_weights(state_dict)

    def synch_value_and_target_weights(self):
        # Copy the weights from action_value network to the target action_value network
        sync_beta = self.options.get('sync_beta', 1.0)
        state_dict = self.q_hat.state_dict().copy()
        target_state_dict = self.q_hat_target.state_dict()
        for name in state_dict:
            state_dict[name] = sync_beta * state_dict[name] + (1.0 - sync_beta) * target_state_dict[name]
        self.set_target_network_weights(state_dict)

    def create_loss_function(self):
        return nn.HuberLoss()

    def create_optimizer(self):
        adam_learning_rate = self.options.get('adam_learning_rate', 0.0001)
        return torch.optim.Adam(self.q_hat.parameters(), lr=adam_learning_rate)

    # TODO : update the getting / setting weights
    def get_value_network_weights(self):
        # TODO : do we need to clone the weights ?
        return self.q_hat.state_dict()

    def set_value_network_weights(self, weights):
        # TODO : do we need the copy?
        self.q_hat.load_state_dict(weights)

    def get_value_network_checksum(self):
        return self.weights_checksum(self.q_hat.state_dict())

    def get_target_network_weights(self):
        return self.q_hat_target.state_dict()

    def set_target_network_weights(self, weights):
        self.q_hat_target.load_state_dict(weights)

    def get_target_network_checksum(self):
        return self.weights_checksum(self.q_hat_target.state_dict())

    def weights_checksum(self, state_dict):
        checksum = 0.0
        for name, layer_weights_or_bias in state_dict.items():
            checksum += layer_weights_or_bias.sum()
        return checksum

    def build_neural_network(self):
        # Create neural network model to predict actions for states.
        try:
            network = QNetwork(self.options)

            # network summary
            print(f"network summary:")
            print(network)

            return network

        except Exception as e:
            print(f"failed to create model : {e}")

    def transpose_states(self, states):
        # states is a 4D array (N, X,Y,Z) with
        # N = number of states,
        # X = state and history, for CNN we need to transpose it to (N, Y,Z,X)
        # and also add another level.
        # not needed in pytorch
        try:
            return np.transpose(np.array(states), (0, 2, 3, 1))
        except Exception as e:
            print(f"transpose failed for : {states}")
            raise e

    def get_all_action_values(self, states):
        return self.q_hat(torch.Tensor(states))

    def get_max_target_values(self, states):
        predictions = self.q_hat_target(torch.Tensor(states))
        return torch.max(predictions, axis=1).values

    def best_action_for(self, state):
        state = torch.Tensor(np.array([state]))
        prediction = self.q_hat(state)
        return torch.argmax(prediction).item()

    def process_batch(self, states, actions, rewards, next_states, terminal):
        """ Process the batch and update the q_func, value function approximation.

        :param states: List of states
        :param actions: List of actions, one for each state
        :param rewards: List of rewards, one for each state
        :param next_states: List of next_states, one for each state
        :param terminal: List of terminal, one for each state
        """
        # If it's not
        not_terminal = torch.IntTensor([1 if not t else 0 for t in terminal])
        rewards = torch.Tensor(rewards)
        # Make predictions for this batch
        predicted_action_values = self.get_all_action_values(np.array(states))
        next_state_action_values = self.get_max_target_values(np.array(next_states))

        y = rewards + not_terminal * self.discount_factor * next_state_action_values
        new_action_values = predicted_action_values.clone()
        # new_action_values is a 2D tensor. Each row is the action values for a state.
        # actions contains a list with the action to be updated for each state (row in the new_action_values tensor)
        new_action_values[torch.arange(new_action_values.size(0)), actions] = y

        # Zero the gradients for every batch!
        self.optimizer.zero_grad()

        # Compute the loss and its gradients
        loss = self.loss_fn(predicted_action_values, new_action_values)
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()


class AgentDqn:

    def __init__(self, options):
        """ Set up the FunctionApprox and policy so that training runs keep using the same.

        :param options: An Options object or Dict containing all the options.
        """
        self.options = Options(options)

        # Set default values
        self.options.default('epsilon', 1.0)
        self.options.default('epsilon_decay_span', 50000)
        self.options.default('epsilon_min', 0.1)
        self.options.default('epsilon_min', 0.1)
        self.options.default('work_dir', options.get('env_name'))

        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        if self.work_dir is not None:
            # location for files.
            self.work_dir = Path(self.work_dir)
            if not self.work_dir.exists():
                self.work_dir.mkdir(parents=True, exist_ok=True)

        self.num_lives = 0

        self.q_func = FunctionApprox(options)
        self.q_func.load_weights()

        self.policy = EGreedyPolicy(self.q_func, options)
        self.play_policy = EGreedyPolicy(self.q_func, {'epsilon': self.options.get('stats_epsilon')})
        # load replay memory data?
        if self.work_dir is None or not self.options.get('load_replay', False):
            replay_memory_file = None
        else:
            replay_memory_file = os.fspath(self.work_dir / 'replay_memory.pickle')
        self.replay_memory = ReplayMemory(max_len=self.options.get('replay_max_size', 100000), history=3,
                                          file_name=replay_memory_file)
        self.max_delta = None
        self.min_delta = None
        self.log_file = self.create_log_file()

    def create_log_file(self):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = self.work_dir / f"episode-detail-{time_stamp}.csv"
        with open(log_file, 'a') as file:
            file.write(f"episode,episode_steps,reward,value_net,target_net,epsilon\n")
        return log_file

    def log_episode_detail(self, episode, steps, reward):
        epsilon = self.policy.epsilon
        print(f"    Episode {episode} : {steps} steps. Epsilon {epsilon:0.3f} : Total reward {reward}")

        q_net_checksum = self.q_func.get_value_network_checksum()
        target_net_checksum = self.q_func.get_target_network_checksum()

        with open(self.log_file, 'a') as file:
            file.write(f"{episode},{steps},{reward}"
                       f",{q_net_checksum},{target_net_checksum},{epsilon}\n")

    def take_step(self, env, action, skip=3):
        previous_obs = None
        life_lost = False
        obs, reward, terminated, truncated, info = env.step(action)
        if 'lives' in info:
            if info['lives'] < self.num_lives:
                self.num_lives = info['lives']
                life_lost = True
        skippy = skip
        while reward == 0 and not terminated and not truncated and not life_lost and skippy > 0:
            skippy -= 1
            previous_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            if 'lives' in info:
                if info['lives'] < self.num_lives:
                    self.num_lives = info['lives']
                    life_lost = True
        # # Try adjusting the reward to penalise losing a life / or the game.
        # if terminated:
        #     reward = -10
        # elif life_lost:
        #     reward = -1

        return self.reformat_observation(obs, previous_obs), reward, terminated, truncated, info, life_lost

    def init_replay_memory(self, env, initial_size):

        while self.replay_memory.size < initial_size:

            obs, info = env.reset()

            state = self.reformat_observation(obs)

            if 'lives' in info:
                # initialise the starting number of lives
                self.num_lives = info['lives']

            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:

                action = self.policy.random_action()
                next_state, reward, terminated, truncated, info, life_lost = self.take_step(env, action)

                self.replay_memory.add(state, action, reward, next_state, terminated or life_lost)
                if self.replay_memory.size >= initial_size:
                    # Replay memory is the required size, so break out.
                    break

                state = next_state

                steps += 1
                if steps >= initial_size:
                    print(f"Break out as we've taken {steps} steps during replay memory initialisation. "
                          f"Something has probably gone wrong...")
                    break

        self.replay_memory.save()

    def play(self, env):
        """ play a single episode using a greedy policy
        :return: Total reward for the game, and a Dict giving the frequency of each action.
        """
        total_reward = 0
        state_with_history = [DataWithHistory.empty_state() for i in range(4)]

        # Init game
        obs, info = env.reset()
        state = self.reformat_observation(obs)
        state_with_history.pop(0)
        state_with_history.append(state)

        if 'lives' in info:
            # initialise the starting number of lives
            self.num_lives = info['lives']

        terminated = False
        truncated = False
        steps = 0
        last_action = -1
        repeated_action_count = 0
        action_frequency = {a: 0 for a in self.q_func.actions}

        while not terminated and not truncated:
            action = self.play_policy.select_action(state_with_history)
            if action == last_action:
                repeated_action_count += 1
                # check it doesn't get stuck
                if repeated_action_count > 1000:
                    print(f"Play with greedy policy has probably got stuck - action {action} repeated 1000 times")
                    break
            else:
                repeated_action_count = 0
            action_frequency[action] += 1
            last_obs = obs

            next_state, reward, terminated, truncated, info, life_lost = self.take_step(env, action)

            total_reward += reward

            state_with_history.pop(0)
            state_with_history.append(next_state)
            last_action = action

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        return total_reward, action_frequency

    def train(self, env, cycle_start, cycle_end):
        target_sync_counter = 0
        agent_rewards = []
        frame_count = 0
        state_with_history = [DataWithHistory.empty_state() for i in range(4)]

        for episode in range(cycle_start, cycle_end):
            # Initialise S
            obs, info = env.reset()

            state = self.reformat_observation(obs)
            state_with_history.pop(0)
            state_with_history.append(state)

            action = self.policy.select_action(state_with_history)

            if 'lives' in info:
                # initialise the starting number of lives
                self.num_lives = info['lives']

            total_undiscounted_reward = 0
            terminated = False
            truncated = False

            self.max_delta = None
            self.min_delta = None

            steps = 0
            actions_before_replay = self.options.get('actions_before_replay', 1)

            while not terminated and not truncated:
                # sync value and target weights based on target_sync_counter option.
                target_sync_counter -= 1
                if target_sync_counter <= 0:
                    self.q_func.synch_value_and_target_weights()
                    target_sync_counter = self.options.get('target_sync_counter', 250)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, info, life_lost = self.take_step(env, action)

                state_with_history.pop(0)
                state_with_history.append(next_state)

                # Choose A' from S' using policy derived from q_func
                next_action = self.policy.select_action(state_with_history)

                self.replay_memory.add(state, action, reward, next_state, terminated or life_lost)
                total_undiscounted_reward += reward

                state, action = next_state, next_action

                # TODO : make the replay steps less frequent?
                # Replay steps from the memory to update the function approximation (q_func)
                actions_before_replay -= 1
                if terminated or actions_before_replay <= 0:
                    self.replay_steps()
                    actions_before_replay = self.options.get('actions_before_replay', 1)

                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

            frame_count += steps
            self.log_episode_detail(episode, steps, total_undiscounted_reward)


            agent_rewards.append(total_undiscounted_reward)

            self.policy.decay_epsilon()

        self.q_func.save_weights()

        self.replay_memory.save()

        return agent_rewards, frame_count

    def replay_steps(self):
        """ Select items from the replay_memory and use them to update the q_func, value function approximation.

        :param replay_num: Number of random steps to replay - currently includes the latest step too.
        """

        states, actions, rewards, next_states, terminal = [], [], [], [], []
        # get the data from the replay memory
        indexes = np.random.randint(0, high=self.replay_memory.size, size=self.options.get('mini_batch_size', 32))
        for i in indexes:
            data_item = self.replay_memory.get_item(i)
            states.append(data_item.get_state())
            actions.append(data_item.get_action())
            rewards.append(data_item.get_reward())
            next_states.append(data_item.get_next_state())
            terminal.append(data_item.is_terminated())
        self.q_func.process_batch(states, actions, rewards, next_states, terminal)


    def reformat_observation(self, obs, previous_obs=None):
        # take the max from obs and last_obs to reduce odd/even flicker that Atari 2600 has
        if previous_obs is not None:
            np.maximum(obs, previous_obs, out=obs)
        # reduce merged greyscalegreyscale from 210,160 down to 84,84
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)


def register_gym_mods():
    """ Register alternatives with gym
    """
    gym.envs.registration.register(
        id='MountainCarMyEasyVersion-v0',
        entry_point='gym.envs.classic_control.mountain_car:MountainCarEnv',
        max_episode_steps=200,      # MountainCar-v0 uses 200
        reward_threshold=-110.0
    )


def run(options):

    options = Options(options)
    options.default('replay_init_size', 50000)
    options.default('episodes', 1)
    options.default('stats_every', 1)

    best_reward = None
    total_steps = 0
    timer = Timer()
    stats = []

    env = gym.make(options.get('env_name'), obs_type='grayscale', render_mode=options.get('render'))

    agent = AgentDqn(options)

    timer.start("Init replay memory")
    agent.init_replay_memory(env, options.get('replay_init_size'))
    timer.stop("Init replay memory")
    replay_init_time = int(timer.event_times["Init replay memory"])
    print(f"Replay memory initialised with {agent.replay_memory.size} items in {replay_init_time} seconds")

    episode_num = options.get('start_episode_num', 1)
    last_episode = episode_num + options.get('episodes') - 1
    cycle_length = options.get('stats_every')
    training_cycles = options.get('episodes') // cycle_length
    if options.get('episodes') % options.get('stats_every') > 0:
        training_cycles += 1

    for i in range(training_cycles):

        timer.start("Training cycle")

        print(f"\nTraining cycle {i+1} of {training_cycles}:")
        # print(f"Start epsilon = {agent.policy.epsilon:0.5f}"
        #       f", discount_factor = {agent.discount_factor}"
        #       f", learning_rate = {agent.adam_learning_rate}")

        cycle_start = options.get('start_episode_num', 1) + i * cycle_length
        cycle_end = min(last_episode + 1, cycle_start + cycle_length)

        rewards, steps = agent.train(env, cycle_start, cycle_end)

        timer.stop("Training cycle")
        training_time = int(timer.event_times["Training cycle"])
        cumulative_training_time = int(timer.cumulative_times["Training cycle"])

        # print some useful info
        total_steps += steps
        print(f"    Steps this cycle = {steps}. "
              f"Total steps all cycles = {total_steps}, in {cumulative_training_time} seconds")
        if len(rewards) == 0:
            max_reward = 0
        else:
            max_reward = max(rewards)
        total_rewards = sum(rewards)
        if best_reward is None:
            best_reward = max_reward
        else:
            best_reward = max(best_reward, max_reward)
        print(f"    Total training rewards from this cycle : {total_rewards}. Best training reward overall = {best_reward}")

        # play the game with greedy policy to get some stats of where we are.
        # Get average for n episodes:
        play_rewards = []
        while len(play_rewards) < 2:
            play_reward, action_frequency = agent.play(env)
            play_rewards.append(play_reward)

        avg_play_reward = sum(play_rewards) / len(play_rewards)
        stats.append((cycle_end-1, avg_play_reward, agent.policy.epsilon))
        print(f"    Avg play reward from current policy at episode {cycle_end-1} = {avg_play_reward:0.2f}")
        plot_reward_epsilon(stats, options, file_only=True)

    env.close()
    del agent

    total_time = timer.cumulative_times["Training cycle"]
    total_mins = int(total_time // 60)
    total_secs = total_time - (total_mins * 60)
    print(f"\nRun finished. Total time taken: {total_mins} mins {total_secs:0.1f} secs")

    # Print the stats out.
    # print(f"Episode  |  Avg Reward")
    # for (episode, reward) in stats:
    #     print(f"{episode:5.0f}   |  {reward:7.2f}")
    plot_reward_epsilon(stats, options)


def plot_reward_epsilon(stats, options, file_only=True):
    # Plot a graph showing reward against episode, and epsilon
    x = [stat[0] for stat in stats]
    rwd = [stat[1] for stat in stats]
    eps = [stat[2] for stat in stats]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_title(options.get('plot_title', 'DQN rewards and epsilon'))
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('reward', color=color)
    # ax1.set_ylim(0, 50)
    ax1.plot(x, rwd, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, eps, color=color)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    plt.savefig(options.get('env_name') + '/' + 'rewards.pdf')
    plt.savefig(options.get('env_name') + '/' + 'rewards.png')
    plt.close('all')


def main():
    # run(render='human')
    options = {
        'work_dir': 'dqn/breakout',
        'env_name': "ALE/Breakout-v5",
        # 'render': "human",
        # 'observation_shape': (84, 84, 4),
        'actions': [0, 1, 2, 3],
        'start_episode_num': 1,
        'episodes': 50,
        'mini_batch_size': 32,
        'actions_before_replay': 4,
        'adam_learning_rate': 0.0001,
        'discount_factor': 0.99,
        'target_sync_counter': 1000,
        # 'sync_beta': 1.0,
        'load_weights': False,
        'save_weights': True,
        'replay_init_size': 50000,
        'replay_max_size': 250000,
        'load_replay': True,
        'stats_epsilon': 0.01,
        'epsilon': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay_episodes': 750,
        'stats_every': 10
    }

    # options['replay_init_size'] = 500
    # options['stats_every'] = 1
    # options['episodes'] = 1
    # options['render'] = 'human'

    # options['start_episode_num'] = 3001
    # options['epsilon'] = 0.1
    # options['adam_learning_rate'] = 0.00001
    # options['env_name'] = "ALE/Pong-v5"
    # options['actions'] = [0, 1, 2, 3, 4, 5]


    run(options)
    # try different options
    # for replay_init_size in [1000, 2000, 4000, 8000]:
    # replay_init_size = 100
    # options['replay_init_size'] = replay_init_size
    # options['replay_max_size'] = replay_init_size*2
    # options['plot_title'] = f"Reward and training epsilon. replay={replay_init_size} : {2*replay_init_size}"

    # # TODO: also decay down to 150 (0.75, 0.25, 150), (0.75, 0.01, 150),
    # for epsilon_values in [(0.75, 0.1, 350), (0.75, 0.1, 2500), (0.01, 0.01, None)]:
    #     options['epsilon'] = epsilon_values[0]
    #     options['epsilon_min'] = epsilon_values[1]
    #     options['epsilon_decay_episodes'] = epsilon_values[2]
    #     options['plot_title'] = f"Reward and training epsilon. Values {epsilon_values}"
    #     options['episodes'] = 2500
    #     run(options)


if __name__ == '__main__':
    main()
