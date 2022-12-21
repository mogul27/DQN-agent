import multiprocessing as mp
from queue import Empty
import os
from pathlib import Path
import time
import random
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt

import gym
import cv2

from dqn_utils import Options, DataWithHistory, Timer, Logger

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim

import numpy as np

import traceback


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
            # We can have multiple workers performing episodes, but each worker bases the decay on their own number
            # of episodes, not the overall number processed by all workers.
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

    def __init__(self, options, q_hat=None, q_hat_target=None):
        self.options = Options(options)

        self.actions = self.options.get('actions')
        if q_hat is None:
            self.q_hat = self.build_neural_network()
        else:
            self.q_hat = q_hat
        if q_hat_target is None:
            self.q_hat_target = self.build_neural_network()
            # set target weights to match q-network
            self.q_hat_target.load_state_dict(self.q_hat.state_dict())
        else:
            self.q_hat_target = q_hat_target
        self.loss_fn = self.create_loss_function()
        self.optimizer = self.create_optimizer()
        self.discount_factor = self.options.get('discount_factor', 0.99)

        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        if self.work_dir is not None:
            # location for files.
            self.work_dir = Path(self.work_dir)
            if not self.work_dir.exists():
                self.work_dir.mkdir(parents=True, exist_ok=True)

        self.log = Logger(self.options.get('log_level', Logger.INFO), 'q_func')

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
                    weights_file.parent().mkdir(parents=True, exist_ok=True)
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

    def get_value_network_weights(self):
        return self.q_hat.state_dict()

    def set_value_network_weights(self, weights):
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
        return np.transpose(np.array(states), (0, 2, 3, 1))

    def get_value(self, state, action):
        # TODO : test if we actually need the transpose.
        prediction = self.q_hat.predict_on_batch(self.transpose_states([state]))
        return prediction[0][action]

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
        return loss.item()


class AsyncQLearnerWorker(mp.Process):

    def __init__(self, global_value_network, global_target_network, messages, grad_update_queue, info_queue,
                 options=None):
        super().__init__()
        self.global_value_network = global_value_network
        self.global_target_network = global_target_network
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.grad_update_queue = grad_update_queue
        self.info_queue = info_queue
        self.options = Options(options)

        # set these up at the start of run. Saves them being pickled/reloaded when the new process starts.
        self.tasks = None
        self.q_func = None
        self.policy = None
        self.initialised = False
        self.num_lives = 0
        self.log = None
        self.value_network_updated = False
        self.grads_sent = 0
        self.worker_throttle = self.options.get('worker_throttle')

    def get_latest_tasks(self, timeout=0.0):
        """ Get the latest tasks from the controller.

        Overrides any existing task of the same name.
        """

        read_messages = True
        while read_messages:
            try:
                msg_code, content = self.messages.get(True, timeout)
                # print(f"got message {msg_code}")
                if msg_code in self.tasks:
                    self.tasks[msg_code] = content
                else:
                    self.log.warn(f"ignored unrecognised message {msg_code}")
            except Empty:
                # Nothing to process, so just carry on
                read_messages = False
            except Exception as e:
                self.log.error(f"read messages failed", e)

    def process_controller_messages(self, timeout=0.0):
        """ see if there are any messages from the controller, and process accordingly
        """
        self.get_latest_tasks(timeout)
        if self.tasks['stop']:
            return

    def get_global_value_network_weights(self):
        """ Controller couldn't keep up with the updates being sent, so add a wait for the network update
        to give the controller chance.
        """
        self.log.trace(f"Get weights from the global value network.")
        self.value_network_updated = False
        while not self.value_network_updated:
            self.process_controller_messages(timeout=0.01)
        self.log.trace(f"Global network weights updated.")

    def async_grad_update(self, experiences):
        """ Process the batch of experiences, calling q_func to apply them.

        :param experiences: List of tuples with state, action, reward, next_state, terminated
        """
        # swap list of tuples to individual lists
        states = []
        next_states = []
        actions = []
        rewards = []
        terminal = []
        for s, a, r, ns, t in experiences:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            terminal.append(t)

        loss = self.q_func.process_batch(states, actions, rewards, next_states, terminal)

        try:
            self.log.trace(f"Sending number of steps to controller {len(experiences)}")
            self.grad_update_queue.put(len(experiences))
        except Exception as e:
            self.log.error(f"failed to send steps to grad_update_queue", e)

        return loss

    def reformat_observation(self, obs, previous_obs=None):
        # take the max from obs and last_obs to reduce odd/even flicker that Atari 2600 has
        if previous_obs is not None:
            np.maximum(obs, previous_obs, out=obs)
        # reduce merged greyscalegreyscale from 210,160 down to 84,84
        resized_obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return resized_obs / 256

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

    def run(self):
        self.log = Logger(self.options.get('log_level', Logger.INFO), f"Worker {self.pid}")
        self.log.info(f"run started")

        self.options.default('async_update_freq', 5)
        self.options.default('discount_factor', 0.99)

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'stop': None
        }

        self.q_func = FunctionApprox(self.options, self.global_value_network, self.global_target_network)

        self.policy = EGreedyPolicy(self.q_func, self.options)
        self.log.info(f"Created policy epsilon={self.policy.epsilon}, "
                      f"min={self.policy.epsilon_min}, decay={self.policy.epsilon_decay}")

        steps = 0
        steps_since_async_update = 0

        env = gym.make(self.options.get('env_name'), obs_type="grayscale")

        action = -1
        total_undiscounted_reward = 0
        experiences = []
        losses = []
        terminated = True   # default to true to get initial reset call

        while True:
            if self.worker_throttle is not None:
                time.sleep(self.worker_throttle)

            self.process_controller_messages()

            if self.tasks['stop']:
                self.log.info(F"stopping due to request from controller")
                return

            if len(experiences) >= self.options.get('async_update_freq'):
                loss = self.async_grad_update(experiences)
                losses.append(loss)
                experiences = []

            if terminated:
                obs, info = env.reset()

                # State includes previous 3 frames.
                state = [DataWithHistory.empty_state() for i in range(3)]
                state.append(self.reformat_observation(obs))

                action = self.policy.select_action(state)

                if 'lives' in info:
                    # initialise the starting number of lives
                    self.num_lives = info['lives']

                total_undiscounted_reward = 0
                terminated = False

            else:
                # take a step and get the next action from the policy.

                # Take action A, observe R, S'
                obs, reward, terminated, truncated, info, life_lost = self.take_step(env, action)
                terminated = terminated or truncated    # treat truncated as terminated

                # take the last 3 frames from state as the history for next_state
                next_state = state[1:]
                next_state.append(obs)  # And add the latest obs

                # Choose A' from S' using policy derived from q_func
                next_action = self.policy.select_action(next_state)

                experiences.append((np.array(state), action, reward, np.array(next_state),
                                            terminated or life_lost))

                steps += 1
                steps_since_async_update += 1

                total_undiscounted_reward += reward
                if terminated:
                    loss = self.async_grad_update(experiences)
                    losses.append(loss)
                    try:
                        # print(f"send info message from worker")
                        # print(f"worker about to send weights deltas to controller")
                        self.info_queue.put({
                            'total_reward': total_undiscounted_reward,
                            'steps': steps,
                            'worker': self.pid,
                            'epsilon': self.policy.epsilon,
                            'value_network_checksum': self.q_func.get_value_network_checksum(),
                            'target_network_checksum': self.q_func.get_target_network_checksum(),
                            'avg_loss': sum(losses) / len(losses)
                        })
                    except Exception as e:
                        self.log.error(f"worker failed to send to info_queue", e)
                    steps = 0
                    # apply epsilon decay after each episode
                    self.policy.decay_epsilon()

                state, action = next_state, next_action


class AsyncStatsCollector(mp.Process):
    """ Gather some stats for the controller when asked to do so.

    Plays a game against the environment using the supplied weights.
    """

    def __init__(self, messages, options=None):
        super().__init__()
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.options = Options(options)

        # set these up at the start of run. Saves them being pickled/reloaded when the new process starts.
        self.tasks = None
        self.q_func = None
        self.policy = None
        self.num_lives = 0
        self.best_reward = None
        self.best_episode = 0
        self.last_n_scores = None
        self.stats = None
        self.work_dir = None
        self.stats_file = None
        self.info_file = None
        self.log = None

    def get_latest_tasks(self):
        """ Get the latest tasks from the controller.

        Overrides any existing task of the same name.
        """

        read_messages = True
        while read_messages:
            try:
                msg_code, content = self.messages.get(False)
                self.log.trace(f"got message {msg_code}")
                if msg_code in self.tasks:
                    self.tasks[msg_code] = content
                else:
                    self.log.warn(f"ignored unrecognised message {msg_code}")
            except Empty:
                # Nothing to process, so just carry on
                read_messages = False
            except Exception as e:
                self.log.error(f"read messages failed", e)

    def process_controller_messages(self, env):
        """ see if there are any messages from the controller, and process accordingly
        """
        self.get_latest_tasks()
        if self.tasks['stop']:
            return

        if self.tasks['play'] is not None:
            contents = self.tasks['play']
            self.tasks['play'] = None
            self.log.trace(f"stats_collector: update weights for play "
                           f"{self.q_func.weights_checksum(contents['weights'])}")
            self.q_func.set_value_network_weights(contents['weights'])

            episode = contents['episode_count']
            play_rewards = []
            while len(play_rewards) < self.options.get('play_avg', 2):
                play_rewards.append(self.play(env, episode))

            avg_play_reward = sum(play_rewards) / len(play_rewards)
            self.stats.append((episode, avg_play_reward))
            with open(self.stats_file, 'a') as file:
                file.write(f"{episode}, {round(avg_play_reward,2):0.2f}\n")
            self.plot_rewards()

    def plot_rewards(self):
        # Plot a graph showing reward against episode, and epsilon
        x = [stat[0] for stat in self.stats]
        rwd = [stat[1] for stat in self.stats]

        fig, ax = plt.subplots()

        color = 'tab:blue'
        ax.set_title(self.options.get('plot_title', 'Asynch Q-Learning'))
        ax.set_xlabel('episodes')
        ax.set_ylabel('reward', color=color)
        # ax1.set_ylim(0, 50)
        ax.plot(x, rwd, color=color)
        ax.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()

        plot_filename = self.options.get('plot_filename', 'async_rewards.png')
        plt.savefig(self.work_dir / plot_filename)
        plt.close('all')    # matplotlib holds onto all the figures if we don't close them.

    def reformat_observation(self, obs, previous_obs=None):
        # take the max from obs and last_obs to reduce odd/even flicker that Atari 2600 has
        if previous_obs is not None:
            np.maximum(obs, previous_obs, out=obs)
        # reduce merged greyscalegreyscale from 210,160 down to 84,84
        resized_obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return resized_obs
        # return resized_obs / 256

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

    def play(self, env, episode_count):
        """ play a single episode using a greedy policy

        :param env: The env to use to run an episode
        :param episode_count: Number of episodes that have been run by the workers to generate the policy
        :return: The reward returned by using the policy against the env.
        """
        # print('stats_collector: play')

        total_reward = 0

        obs, info = env.reset()

        # State includes previous 3 frames.
        state = [DataWithHistory.empty_state() for i in range(3)]
        state.append(self.reformat_observation(obs))

        if 'lives' in info:
            # initialise the starting number of lives
            self.num_lives = info['lives']

        terminated = False
        steps = 0
        last_action = -1
        repeated_action_count = 0
        action_frequency = {a: 0 for a in self.options.get('actions')}
        life_lost = True

        while not terminated:
            if life_lost:
                # assume game starts with action 1 - Fire.
                action = 1
            else:
                action = self.policy.select_action(state)
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

            obs, reward, terminated, truncated, info, life_lost = self.take_step(env, action)
            total_reward += reward

            terminated = terminated or truncated    # treat truncated as terminated

            # remove the oldest frame from the state
            state.pop(0)
            state.append(obs)  # And add the latest obs

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        if self.best_reward is None or total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_episode = episode_count

        self.last_n_scores.append(total_reward)

        self.log.info(f"Play : weights from episode {episode_count}"
                      f" : Reward = {total_reward}"
                      f" : epsilon: {self.policy.epsilon:0.4f}"
                      f" : value_net: {self.q_func.get_value_network_checksum():0.4f}"
                      f" : Action freq: {action_frequency}"
                      f" : Best episode = {self.best_episode} : Best reward = {self.best_reward}"
                      f" : Avg reward (last {len(self.last_n_scores)}) = {mean(self.last_n_scores)}")

        with open(self.info_file, 'a') as file:
            file.write(f'{episode_count}, {total_reward}, {self.policy.epsilon}, "{action_frequency}"\n')

        return total_reward

    def run(self):
        self.log = Logger(self.options.get('log_level', Logger.INFO), f'Stats {self.pid}')
        self.log.info(f"started run")

        self.options.default('stats_epsilon', 0.05)
        self.last_n_scores = deque(maxlen=5)
        self.stats = []
        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        # location for files.
        self.work_dir = Path(self.work_dir)
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.stats_file = self.work_dir / f"rewards-{time_stamp}.csv"
        with open(self.stats_file, 'a') as file:
            file.write(f"episode, reward\n")
        self.info_file = self.work_dir / f"info-{time_stamp}.csv"
        with open(self.info_file, 'a') as file:
            file.write(f"episode, reward, epsilon, value_net_checksum, action_frequency\n")

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'play': None,
            'stop': None
        }

        self.q_func = FunctionApprox(self.options)
        # use a policy with epsilon of 0.05 for playing to collect stats.
        self.policy = EGreedyPolicy(self.q_func, {'epsilon': self.options.get('stats_epsilon')})
        print(f"stats collector {self.pid}: Created policy epsilon={self.policy.epsilon}, "
              f"min={self.policy.epsilon_min}, decay={self.policy.epsilon_decay}")

        env = gym.make(self.options.get('env_name'), obs_type="grayscale")

        while True:

            self.process_controller_messages(env)

            if self.tasks['stop']:
                print(F"stats collector {self.pid} stopping due to request from controller")
                return


class AsyncQLearningController:

    def __init__(self, options=None):
        self.options = Options(options)
        self.q_func = FunctionApprox(self.options)
        self.global_value_network = self.q_func.q_hat
        self.global_target_network = self.q_func.q_hat_target
        self.global_value_network.share_memory()
        self.global_target_network.share_memory()
        self.q_func.load_weights()
        self.workers = []
        self.stats_collector = None
        self.delta_fraction = self.options.get('delta_fraction', 1.0)
        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        # location for files.
        self.work_dir = Path(self.work_dir)
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.create_controller_log()
        self.log = Logger(self.options.get('log_level', Logger.INFO), "Controller")

    def message_all_workers(self, msg_code, content):
        msg = (msg_code, content)
        for worker, worker_queue in self.workers:
            # send message to each worker
            self.log.trace(f"sending : {msg_code}")
            try:
                worker_queue.put(msg)
            except Exception as e:
                self.log.error(f"Queue put failed in controller", e)
            self.log.trace(f"sent : {msg_code}")

    def message_stats_collector(self, msg_code, content):
        msg = (msg_code, content)
        stats_queue, worker = self.stats_collector
        try:
            self.log.trace(f"Send {msg_code} message to stats collector")
            stats_queue.put(msg)
        except Exception as e:
            self.log.error(f"stats_queue put failed in controller", e)

    def create_controller_log(self):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = self.work_dir / f"episode-detail-{time_stamp}.csv"
        with open(log_file, 'a') as file:
            file.write(f"episode, total_steps, reward, pid, value_net, target_net, epsilon\n")
        return log_file

    def log_episode_detail(self, episode_detail, total_steps, episode):
        if self.options.get('log_level', 2) < 2:
            print(f"Total steps {total_steps} : Episode {episode} took {episode_detail['steps']} steps"
                  f" : reward = {episode_detail['total_reward']} : pid = {episode_detail['worker']}"
                  f" : value_net = {episode_detail['value_network_checksum']:0.4f}"
                  f" : target_net = {episode_detail['target_network_checksum']:0.4f}"
                  f" : epsilon = {episode_detail['epsilon']:0.4f}")

        with open(self.log_file, 'a') as file:
            file.write(f"{episode},{total_steps}"
                       f",{episode_detail['total_reward']}"
                       f",{episode_detail['worker']}"
                       f",{episode_detail['value_network_checksum']}"
                       f",{episode_detail['target_network_checksum']}"
                       f",{episode_detail['epsilon']}"
                       f"\n")

    def train(self):
        print(f"{os.getppid()}: Setting up workers to run asynch q-learning")

        grad_update_queue = mp.Queue()
        info_queue = mp.Queue()

        # set up the workers

        for _ in range(self.options.get('num_workers', 1)):
            worker_queue = mp.Queue()
            worker = AsyncQLearnerWorker(self.global_value_network, self.global_target_network,
                worker_queue, grad_update_queue, info_queue, self.options)
            worker.daemon = True    # helps tidy up child processes if parent dies.
            worker.start()
            self.workers.append((worker, worker_queue))

        # set up stats_gatherer
        stats_queue = mp.Queue()
        worker = AsyncStatsCollector(stats_queue, self.options)
        worker.daemon = True    # helps tidy up child processes if parent dies.
        worker.start()
        self.stats_collector = (stats_queue, worker)

        total_steps = 0
        episode_count = 0
        target_sync_counter = self.options.get('target_net_sync_steps', 1)
        grad_updates = 0
        grads_update_errors = 0

        while episode_count < self.options.get('episodes', 1):

            grad_update_messages = []
            try:
                while True:
                    grad_update_messages.append(grad_update_queue.get(False))
                    grad_updates += 1
                    self.log.trace(f"got network gradients. Total received={grad_updates}, "
                                   f"error count={grads_update_errors}")

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                grads_update_errors += 1
                self.log.error(f"Failed to get gradients off the queue. Total received={grad_updates}, "
                               f"error count={grads_update_errors}", e)

            if len(grad_update_messages) > 0:
                self.log.trace(f"accumulate steps from {len(grad_update_messages)} messages")
                for steps in grad_update_messages:
                    total_steps += steps
                    target_sync_counter -= steps

            if target_sync_counter <= 0:
                # synch the target weights with the value weights, then let the workers know.
                self.q_func.synch_value_and_target_weights()
                target_sync_counter = self.options.get('target_net_sync_steps', 1)

            try:
                self.log.trace(f"about to read from info queue")
                info = info_queue.get(False)
                episode_count += 1
                self.log_episode_detail(info, total_steps, episode_count)

                if episode_count % self.options.get('stats_every') == 0:
                    self.q_func.save_weights()
                    weights = self.q_func.get_value_network_weights().copy()
                    self.message_stats_collector('play', {'weights': weights, 'episode_count': episode_count})

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                self.log.error(f"Failed to read info_queue", e)

        # close down the workers
        print(f"All done, {total_steps} steps processed - close down the workers")
        self.message_all_workers('stop', True)
        self.message_stats_collector('stop', True)

        self.q_func.save_weights()

        time.sleep(5)   # give them a chance to stop
        try:
            # make sure they've all stopped OK
            for worker, _ in self.workers:
                worker.terminate()
                worker.join(5)
            stats_queue, stats_worker = self.stats_collector
            stats_worker.terminate()
            stats_worker.join(5)
        except Exception as e:
            self.log.error("Failed to close the workers cleanly", e)


def create_and_run_agent(options):
    timer = Timer()
    timer.start("agent")

    agent = AsyncQLearningController(options)
    agent.train()

    timer.stop('agent')
    elapsed = timer.event_times['agent']
    hours = int(elapsed / 3600)
    elapsed = elapsed - 3600*hours
    mins = int(elapsed / 60)
    secs = int(elapsed - 60*mins)
    print(f"Total time taken by agent is {hours} hours {mins} mins {secs} secs")


if __name__ == '__main__':
    print("Use spawn - should work on all op sys.")
    mp.set_start_method('spawn')

    options = {
        'work_dir': 'async/breakout',
        'env_name': "ALE/Breakout-v5",
        'actions': [0, 1, 2, 3],
        # 'work_dir': 'async/pong',
        # 'env_name': "ALE/Pong-v5",
        # 'actions': [0, 1, 2, 3, 4, 5],
        # 'render': "human",
        'actions': [0, 1, 2, 3],
        'num_workers': 4,
        'episodes': 8000,
        'async_update_freq': 5,
        'target_net_sync_steps': 4000,
        'sync_beta': 1.0,
        'adam_learning_rate': 0.0001,
        'discount_factor': 0.99,
        'load_weights': False,
        'save_weights': True,
        'stats_epsilon': 0.01,
        'epsilon': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay_episodes': 1000,
        'stats_every': 20,  # how frequently to collect stats
        'play_avg': 1,      # number of games to average
        'log_level': Logger.INFO,     # debug=1, info=2, warn=3, error=4
        'worker_throttle': 0.005,       # 0.001 a bit low for 2 workers, and 0.005 or 0.01 for 8
    }

    # for lr in [0.01, 0.001, 0.0001]:
    #     for num_workers in [1, 2, 4]:
    #         options['plot_filename'] = f'async_rewards_w_{num_workers}_lr_{lr}.png'
    #         options['plot_title'] = f"Asynch Q-Learning {num_workers} workers, lr={lr}"
    #         options['num_workers'] = num_workers
    #         options['adam_learning_rate'] = lr
    #         create_and_run_agent(options)

    options['plot_filename'] = f'async_rewards_breakout.png'
    options['plot_title'] = f"Asynch Q-Learning Breakout 8 workers"
    create_and_run_agent(options)
