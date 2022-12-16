# Try and do the gym's atari breakout

import gym
import numpy as np
import random
import os
from pathlib import Path
import pickle
from collections import deque
import matplotlib.pyplot as plt

from keras import Model
from keras.layers import Dense, Input
from keras.losses import Huber
from keras.optimizers import Adam
from dqn_utils import Timer, Options


class EGreedyPolicy:
    """ Assumes every state has the same possible actions.
    """

    def __init__(self, q_func, options=None):
        """ e-greedy policy based on the supplied q_table.

        :param q_func: Approximates q values for state action pairs so we can select the best action.
        :param options: can contain:
            'epsilon': small epsilon for the e-greedy policy. This is the probability that we'll
                       randomly select an action, rather than picking the best.
            'epsilon_decay_span': the number of calls over which to decay epsilon
            'epsilon_min': the min value epsilon can be after decay
        """
        self.options = Options(options)
        self.options.default('epsilon', 0.1)

        self.q_func = q_func
        self.epsilon = self.options.get('epsilon')
        self.possible_actions = self.q_func.actions
        if self.options.get('epsilon_decay_episodes') is None:
            self.epsilon_min = self.epsilon
            self.epsilon_decay = 0
        else:
            if self.options.get('epsilon_min') is None:
                self.epsilon_min = 0
            else:
                self.epsilon_min = self.options.get('epsilon_min')
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.options.get('epsilon_decay_episodes')

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
        # decay epsilon for next time
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)


class FunctionApprox:

    def __init__(self, options):
        self.options = Options(options)

        self.actions = self.options.get('actions')
        self.q_hat = self.build_neural_network()
        self.q_hat_target = self.build_neural_network()
        self.synch_value_and_target_weights()
        self.batch = []

    def save_weights(self, file_name):
        self.q_hat.save_weights(file_name)

    def load_weights(self, file_name):
        if Path(file_name).exists():
            self.q_hat.load_weights(file_name)
            self.q_hat_target.load_weights(file_name)

    def synch_value_and_target_weights(self):
        # Copy the weights from action_value network to the target action_value network
        value_weights = self.q_hat.get_weights()
        target_weights = self.q_hat_target.get_weights()
        # TODO : consider moving towards the value weights, rather than a complete replacement.
        sync_fraction = self.options.get('sync_fraction', 1.0)
        for i in range(len(target_weights)):
            target_weights[i] = (1 - sync_fraction) * target_weights[i] + sync_fraction * value_weights[i]
        self.q_hat_target.set_weights(target_weights)

    def build_neural_network(self):
        # Crete neural network model to predict actions for states.
        try:
            # TODO : give all the layers and models names to indicate worker / controller ?

            inputs = Input(self.options.get('observation_shape'))
            dense_1 = Dense(64, activation='relu')(inputs)
            dense_2 = Dense(64, activation='relu')(dense_1)
            outputs = Dense(len(self.actions), activation='linear')(dense_2)
            network = Model(inputs=inputs, outputs=outputs)

            network.summary()

            # compile the model
            adam_learning_rate = self.options.get('adam_learning_rate', 0.0001)
            network.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=adam_learning_rate))

            return network
        except Exception as e:
            print(f"failed to create model : {e}")

    def get_value(self, state, action):
        prediction = self.q_hat.predict_on_batch(np.expand_dims(state, axis=0))
        return prediction[0][action]

    def get_all_action_values(self, states):
        return self.q_hat.predict_on_batch(states)

    def get_target_value(self, state, action):
        prediction = self.q_hat_target.predict_on_batch(np.expand_dims(state, axis=0))
        return prediction[0][action]

    def get_max_target_value(self, state):
        prediction = self.q_hat_target.predict_on_batch(np.expand_dims(state, axis=0))
        return max(prediction[0])

    def get_max_target_values(self, states):
        predictions = self.q_hat_target.predict_on_batch(states)
        return predictions.max(axis=1)

    def best_action_for(self, state):
        prediction = self.q_hat.predict_on_batch(np.expand_dims(state, axis=0))
        return np.argmax(prediction[0])

    def update_batch(self, batch):
        # do the update in batches
        states = np.array([s for (s, new_action_value) in batch])
        new_action_values = np.array([new_action_value for (s, new_action_value) in batch])

        return self.q_hat.train_on_batch(states, new_action_values)


class AgentDqn:

    def __init__(self, options):
        """ Set up the FunctionApprox and policy so that training runs keep using the same.

        :param load_weights:
        :param exploratory_action_probability:
        """
        self.options = Options(options)

        # Set default values
        self.options.default('epsilon', 1.0)
        self.options.default('epsilon_decay_span', 50000)
        self.options.default('epsilon_min', 0.1)
        self.options.default('epsilon_min', 0.1)
        self.options.default('work_dir', options.get('env_name'))

        self.work_dir = self.options.get('work_dir')
        if self.options.get('work_dir') is None:
            self.work_dir = None
        else:
            # location for files.
            self.work_dir = Path(self.options.get('work_dir'))
            if not self.work_dir.exists():
                self.work_dir.mkdir(parents=True, exist_ok=True)

        self.discount_factor = self.options.get('discount_factor', 0.99)

        self.num_lives = 0

        possible_actions = [0, 1, 2]

        self.q_func = FunctionApprox(options)
        # if load_weights is not None:
        #     if self.work_dir is not None:
        #         load_weights = os.fspath(self.work_dir / load_weights)
        #     self.q_func.load_weights(load_weights)
        self.policy = EGreedyPolicy(self.q_func, options)
        self.play_policy = EGreedyPolicy(self.q_func, {'epsilon': self.options.get('stats_epsilon')})
        # TODO : load replay memory data?
        # if self.work_dir is None:
        #     replay_memory_file = None
        # else:
        #     replay_memory_file = os.fspath(self.work_dir / 'replay_memory.pickle')
        self.replay_memory = deque(maxlen=self.options.get('replay_max_size', 100000))
        self.max_delta = None
        self.min_delta = None
        self.total_episodes = 0

    def load_replay_memory(self):
        if self.work_dir is not None:
            replay_memory_file = Path(self.work_dir / 'replay_memory.pickle')
            if os.path.isfile(replay_memory_file):
                with open(replay_memory_file, 'rb') as file:
                    self.replay_memory = pickle.load(file)

    def save_replay_memory(self):
        if self.work_dir is not None:
            replay_memory_file = Path(self.work_dir / 'replay_memory.pickle')
            with open(replay_memory_file, 'wb') as file:
                pickle.dump(self.replay_memory, file)

    def init_replay_memory(self, env, initial_size):
        self.load_replay_memory()

        initial_size = min(initial_size, self.replay_memory.maxlen)
        while len(self.replay_memory) < initial_size:

            state, info = env.reset()

            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:

                action = self.policy.random_action()
                next_state, reward, terminated, truncated, info = env.step(action)

                self.replay_memory.append((state, action, reward, next_state, terminated or truncated))
                if len(self.replay_memory) >= initial_size:
                    # Replay memory is the required size, so break out.
                    break

                state = next_state

                steps += 1
                if steps >= initial_size:
                    print(f"Break out as we've taken {steps} steps during replay memory initialisation. "
                          f"Something has probably gone wrong...")
                    break

        self.save_replay_memory()

    def play(self, env):
        """ play a single episode using a greedy policy
        :return: Total reward for the game, and a Dict giving the frequency of each action.
        """
        total_reward = 0

        # Init game
        state, info = env.reset()

        terminated = False
        truncated = False
        steps = 0
        last_action = -1
        repeated_action_count = 0
        action_frequency = {a: 0 for a in self.q_func.actions}

        while not terminated and not truncated:
            action = self.play_policy.select_action(state)
            if action == last_action:
                repeated_action_count += 1
                # check it doesn't get stuck
                if repeated_action_count > 1000:
                    print(f"Play with greedy policy has probably got stuck - action {action} repeated 1000 times")
                    break
            else:
                repeated_action_count = 0
            action_frequency[action] += 1

            state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            last_action = action

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        return total_reward, action_frequency

    def train(self, env, cycle_start, cycle_end):
        sync_weights_count = 0
        agent_rewards = []
        frame_count = 0

        for episode in range(cycle_start, cycle_end):
            # Initialise S
            state, info = env.reset()

            action = self.policy.select_action(state)

            total_undiscounted_reward = 0
            terminated = False
            truncated = False

            self.max_delta = None
            self.min_delta = None

            steps = 0
            actions_before_replay = 4

            while not terminated and not truncated:
                # sync value and target weights at the start of an episode
                sync_weights_count -= 1
                if sync_weights_count <= 0:
                    self.q_func.synch_value_and_target_weights()
                    sync_weights_count = self.options.get('sync_weights_count', 250)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, info = env.step(action)

                # Choose A' from S' using policy derived from q_func
                next_action = self.policy.select_action(state)

                self.replay_memory.append((state, action, reward, next_state, terminated))
                total_undiscounted_reward += reward

                state, action = next_state, next_action

                # TODO : make the replay steps less frequent?
                # Replay steps from the memory to update the function approximation (q_func)
                actions_before_replay -= 1
                if terminated or actions_before_replay <= 0:
                    self.replay_steps()
                    actions_before_replay = 4

                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

            self.total_episodes += 1
            print(f"    Episode {self.total_episodes} : {steps} steps. Epsilon {self.policy.epsilon:0.3f}"
                  f" : Total reward {total_undiscounted_reward}")
            frame_count += steps
            agent_rewards.append(total_undiscounted_reward)

            self.policy.decay_epsilon()

        # TODO : save the weights?
        # if save_weights is not None:
        #     if self.work_dir is not None:
        #         self.q_func.save_weights(self.work_dir / save_weights)
        #     else:
        #         self.q_func.save_weights(save_weights)
        #
        # TODO : should we save the replay_memory here?
        #        Might be useful if we're restarting training
        # self.save_replay_memory()

        return agent_rewards, frame_count

    def process_batch(self, state_action_batch):
        """ Process the batch and update the q_func, value function approximation.

        :param state_action_batch: List of tuples with state, action, reward, next_state, terminated
        """
        # process the batch - get the values and target values in single calls
        # TODO : make this neater - vector based?
        states = []
        next_states = []
        actions = []
        rewards = []
        for data_item in state_action_batch:
            states.append(data_item[0])
            next_states.append(data_item[3])
            actions.append(data_item[1])
            rewards.append(data_item[2])
        states = np.array(states)
        next_states = np.array(next_states)

        qsa_action_values = self.q_func.get_all_action_values(states)
        next_state_action_values = self.q_func.get_max_target_values(next_states)
        discounted_next_qsa_values = self.discount_factor * next_state_action_values
        updates = []
        min_d = None
        max_d = None
        min_q = None
        max_q = None
        # weights = self.q_func.get_value_network_weights()
        # print(f"worker {self.pid} : weights[0][0][0][0][0:5] {weights[0][0][0][0][0:5]}")
        # print(f"worker {self.pid} : weights[1][0] {weights[1][0]}")
        #
        # print(f"worker {self.pid} : a={actions} : r={rewards} : qsa={qsa_action_values} : qsa_next={discounted_next_qsa_values}")
        for data_item, qsa_action_value, discounted_next_qsa_value in zip(state_action_batch, qsa_action_values, discounted_next_qsa_values):
            s, a, r, next_s, terminated = data_item
            if terminated:
                y = r
            else:
                y = r + discounted_next_qsa_value

            delta = qsa_action_value[a] - y
            if min_d is None:
                min_d = delta
                max_d = delta
            else:
                min_d = min(delta, min_d)
                max_d = max(delta, max_d)

            if min_q is None:
                min_q = qsa_action_value[a]
                max_q = qsa_action_value[a]
            else:
                min_q = min(qsa_action_value[a], min_q)
                max_q = max(qsa_action_value[a], max_q)

            # update the action value to move closer to the target
            qsa_action_value[a] = y

            updates.append((s, qsa_action_value))

        losses = self.q_func.update_batch(updates)
        return losses, min_d, max_d, min_q, max_q

    def replay_steps(self, replay_num=32):
        """ Select items from the replay_memory and use them to update the q_func, value function approximation.

        :param replay_num: Number of random steps to replay - currently includes the latest step too.
        """
        indexes = np.random.randint(0, high=len(self.replay_memory), size=replay_num)
        batch = [self.replay_memory[i] for i in indexes]
        loss, min_d, max_d, min_q, max_q = self.process_batch(batch)
        # if min_delta is None:
        #     min_delta = min_d
        #     max_delta = max_d
        # else:
        #     min_delta = min(min_d, min_delta)
        #     max_delta = min(max_d, max_delta)
        # if min_q_value is None:
        #     min_q_value = min_q
        #     max_q_value = max_q
        # else:
        #     min_q_value = min(min_q, min_q_value)
        #     max_q_value = min(max_q, max_q_value)


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

    env = gym.make(options.get('env_name'), render_mode=options.get('render'))

    agent = AgentDqn(options)

    timer.start("Init replay memory")
    agent.init_replay_memory(env, options.get('replay_init_size'))
    timer.stop("Init replay memory")
    replay_init_time = int(timer.event_times["Init replay memory"])
    print(f"Replay memory initialised with {len(agent.replay_memory)} items in {replay_init_time} seconds")

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
        while len(play_rewards) < 5:
            play_reward, action_frequency = agent.play(env)
            play_rewards.append(play_reward)

        avg_play_reward = sum(play_rewards) / len(play_rewards)
        stats.append((cycle_end-1, avg_play_reward, agent.policy.epsilon))
        print(f"    Avg play reward from current policy at episode {cycle_end-1} = {avg_play_reward:0.2f}")

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


def plot_reward_epsilon(stats, options):
    # Plot a graph showing reward against episode, and epsilon
    x = [stat[0] for stat in stats]
    rwd = [stat[1] for stat in stats]
    eps = [stat[2] for stat in stats]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_title(options.get('plot_title', 'DQN rewards and epsilon'))
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('reward', color=color)
    ax1.set_ylim(0, 500)
    ax1.plot(x, rwd, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, eps, color=color)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    plt.show()


def main():
    # run(render='human')
    options={
        'env_name': "CartPole-v1",
        # 'render': "human",
        'observation_shape': (4, ),
        'actions': [0, 1],
        'episodes': 1000,
        'adam_learning_rate': 0.001,
        'discount_factor': 0.85,
        'sync_weights_count': 400,
        'sync_fraction': 1.0,
        'load_weights': False,
        'save_weights': False,
        'replay_init_size': 1000,
        'replay_max_size': 2000,
        'stats_epsilon': 0.01,
        'epsilon': 0.75,
        'epsilon_min': 0.1,
        'epsilon_decay_episodes': 350,
        'stats_every': 10
    }

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
