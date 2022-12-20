import torch.multiprocessing as mp
from queue import Empty
import os
from pathlib import Path
import time
import random
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt

import gym

from dqn_utils import Options, Timer

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim

from copy import deepcopy

import numpy as np

import traceback


class Logger:
    """ basic logger """
    ERROR = 'ERROR'
    WARN = 'WARN'
    INFO = 'INFO'
    DEBUG = 'DEBUG'

    def __init__(self, log_level, name=""):
        self.log_level = log_level
        self.name = name

    def print(self, level, message):
        print(f"{level:6}: {self.name} : {message}")

    def debug(self, message):
        if self.log_level <= 1:
            self.print(Logger.DEBUG, message)

    def info(self, message):
        if self.log_level <= 2:
            self.print(Logger.INFO, message)

    def warn(self, message):
        if self.log_level <= 3:
            self.print(Logger.WARN, message)

    def error(self, message, e=None):
        stack_trace = "\n".join(traceback.format_stack()[-6:-1])
        if self.log_level <= 4:
            self.print(Logger.ERROR, f"{message}\n{type(e)}: {e}\n{stack_trace}")


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
        self.options = Options(options)
        self.fc1 = nn.Linear(self.options.get('observation_shape')[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(self.options.get('actions')))

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FunctionApprox:

    def __init__(self, options):
        self.options = Options(options)

        self.actions = self.options.get('actions')
        self.q_hat = self.build_neural_network()
        self.q_hat_target = self.build_neural_network()
        # set target weights to match q-network
        self.q_hat_target.load_state_dict(self.q_hat.state_dict())
        self.batch = []
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
        # Crete neural network model to predict actions for states.
        try:
            network = QNetwork(self.options)

            # network summary
            print(f"network summary:")
            print(network)

            # compile the model?

            return network
        except Exception as e:
            print(f"failed to create model : {e}")

    def get_all_action_values(self, states):
        return self.q_hat(torch.Tensor(states))

    def get_max_target_values(self, states):
        predictions = self.q_hat_target(torch.Tensor(states))
        return torch.max(predictions, axis=1).values

    def best_action_for(self, state):
        prediction = self.q_hat(torch.Tensor(state))
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

        # Adjust learning weights - not doing this in the worker, simply get the gradients
        # self.optimizer.step()

    def apply_grads(self, parameter_grads):
        self.optimizer.zero_grad()
        # loss.backward()
        for worker_param_grad, global_param in zip(parameter_grads, self.q_hat.parameters()):
            global_param._grad = worker_param_grad
        self.optimizer.step()


class AsyncQLearnerWorker(mp.Process):

    def __init__(self, messages, network_gradient_queue, info_queue, options=None):
        super().__init__()
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.network_gradient_queue = network_gradient_queue
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
                    print(f"ignored unrecognised message {msg_code}")
            except Empty:
                # Nothing to process, so just carry on
                read_messages = False
            except Exception as e:
                print(f"read messages failed")
                print(e)

    def process_controller_messages(self, timeout=0.0):
        """ see if there are any messages from the controller, and process accordingly
        """
        self.get_latest_tasks(timeout)
        if self.tasks['stop']:
            return

        if self.tasks['value_network_weights'] is not None:
            # print(f"update value network weights "
            #       f"{self.q_func.weights_checksum(self.tasks['value_network_weights'])}")
            self.log.debug(f"update value network weights")
            network_weights = self.tasks['value_network_weights']
            self.q_func.set_value_network_weights(network_weights)
            self.tasks['value_network_weights'] = None
            self.value_network_updated = True

        if self.tasks['target_network_weights'] is not None:
            self.log.debug(f"update target network weights")
            # print(f"update target network weights "
            #       f"{self.q_func.weights_checksum(self.tasks['target_network_weights'])}")
            self.q_func.set_target_network_weights(self.tasks['target_network_weights'])
            self.tasks['target_network_weights'] = None

        if self.tasks['reset_network_weights'] is not None:
            self.log.debug(f"reset_network_weights network weights")
            network_weights = self.tasks['reset_network_weights']
            # print(f"reset value and target network weights "
            #       f"{self.q_func.weights_checksum(network_weights)}")
            self.q_func.set_value_network_weights(network_weights)
            self.q_func.set_target_network_weights(network_weights)

            self.initialised = True
            self.tasks['reset_network_weights'] = None

    def get_global_value_network_weights(self):
        """ Controller couldn't keep up with the updates being sent, so add a wait for the network update
        to give the controller chance.
        """
        self.log.debug(f"Get weights from the global value network.")
        self.value_network_updated = False
        while not self.value_network_updated:
            self.process_controller_messages(timeout=0.001)
        self.log.debug(f"Global network weights updated.")

    def process_experiences(self, experiences):
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

        self.q_func.process_batch(states, actions, rewards, next_states, terminal)

    def async_grad_update(self, experiences):
        # We've processed enough steps to do an update.
        self.process_experiences(experiences)

        # get all the parameters with their gradients
        parameter_grads = []
        for param in self.q_func.q_hat.parameters():
            parameter_grads.append(param.grad.clone())

        try:
            self.log.debug(f"Sending gradients to controller. steps {len(experiences)}")
            self.network_gradient_queue.put((len(experiences), parameter_grads))
            self.grads_sent += 1
            self.log.debug(f"gradients sent. Total sent since start : {self.grads_sent}")
            self.get_global_value_network_weights()
        except Exception as e:
            self.log.error(f"failed to send gradients", e)

        return parameter_grads

    def run(self):
        self.log = Logger(self.options.get('log_level', 2), f"Worker {self.pid}")
        self.log.info(f"run started")

        self.options.default('async_update_freq', 5)
        self.options.default('discount_factor', 0.99)

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'value_network_weights': None,
            'target_network_weights': None,
            'reset_network_weights': None,
            'stop': None
        }

        self.q_func = FunctionApprox(self.options)

        # TODO : add epsilon to options
        self.policy = EGreedyPolicy(self.q_func, self.options)
        print(f"Created policy epsilon={self.policy.epsilon}, "
              f"min={self.policy.epsilon_min}, decay={self.policy.epsilon_decay}")

        self.initialised = False

        steps = 0
        steps_since_async_update = 0

        register_gym_mods()
        env = gym.make(self.options.get('env_name'))
        state = None
        terminated = True
        action = -1
        total_undiscounted_reward = 0
        experiences = []
        parameter_grads = None

        while True:

            self.process_controller_messages()

            if self.tasks['stop']:
                self.log.info(F"stopping due to request from controller")
                return

            if len(experiences) >= self.options.get('async_update_freq'):
                parameter_grads = self.async_grad_update(experiences)
                experiences = []

            if self.initialised:
                if terminated:
                    state, info = env.reset()

                    action = self.policy.select_action(state)

                    total_undiscounted_reward = 0
                    terminated = False

                else:
                    # Take action A, observe R, S'
                    next_state, reward, terminated, truncated, info = env.step(action)
                    terminated = terminated or truncated    # treat truncated as terminated

                    # Choose A' from S' using policy derived from q_func
                    next_action = self.policy.select_action(next_state)

                    experiences.append((state, action, reward, next_state, terminated))

                    steps += 1
                    steps_since_async_update += 1

                    total_undiscounted_reward += reward
                    if terminated:
                        # TODO : add a calculate and send of the grads at the end of an episode.
                        parameter_grads = self.async_grad_update(experiences)

                        try:
                            # print(f"send info message from worker")
                            # print(f"worker about to send weights deltas to controller")
                            self.info_queue.put({
                                'total_reward': total_undiscounted_reward,
                                'steps': steps,
                                'worker': self.pid,
                                'epsilon': self.policy.epsilon,
                                'value_network_checksum': self.q_func.get_value_network_checksum(),
                                'target_network_checksum': self.q_func.get_target_network_checksum()
                            })
                        except Exception as e:
                            print(f"worker failed to send weight deltas")
                            print(e)
                        # print(f"Finished episode after {steps} steps. "
                        #       f"Total reward {total_undiscounted_reward}")
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
                self.log.debug(f"got message {msg_code}")
                if msg_code in self.tasks:
                    self.tasks[msg_code] = content
                else:
                    self.log.warn(f"ignored unrecognised message {msg_code}")
            except Empty:
                # Nothing to process, so just carry on
                read_messages = False
            except Exception as e:
                print(f"read messages failed")
                print(e)

    def process_controller_messages(self, env):
        """ see if there are any messages from the controller, and process accordingly
        """
        self.get_latest_tasks()
        if self.tasks['stop']:
            return

        if self.tasks['play'] is not None:
            # print(f"update the value network weights")
            contents = self.tasks['play']
            self.tasks['play'] = None
            # print(f"stats_collector: update weights play  {self.q_func.weights_checksum(contents['weights'])}")
            self.q_func.set_value_network_weights(contents['weights'])
            updated_weights = self.q_func.get_value_network_weights()
            # print(f"stats_collector: updated weights play  {self.q_func.weights_checksum(updated_weights)}")

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
        ax.set_title(self.options.get('plot_title', 'Asynch DQN rewards'))
        ax.set_xlabel('episodes')
        ax.set_ylabel('reward', color=color)
        # ax1.set_ylim(0, 50)
        ax.plot(x, rwd, color=color)
        ax.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()

        plt.savefig(self.work_dir / 'async_rewards.pdf')
        plt.savefig(self.work_dir / 'async_rewards.png')
        plt.close('all')    # matplotlib holds onto all the figures if we don't close them.

    def play(self, env, episode_count):
        """ play a single episode using a greedy policy

        :param env: The env to use to run an episode
        :param episode_count: Number of episodes that have been run by the workers to generate the policy
        :return: The reward returned by using the policy against the env.
        """
        # print('stats_collector: play')

        total_reward = 0

        state, info = env.reset()

        terminated = False
        steps = 0
        last_action = -1
        repeated_action_count = 0
        action_frequency = {a: 0 for a in self.options.get('actions')}

        while not terminated:
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

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            terminated = terminated or truncated    # treat truncated as terminated

            steps += 1
            if steps >= 100000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        if self.best_reward is None or total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_episode = episode_count

        self.last_n_scores.append(total_reward)

        print(f"Play : weights from episode {episode_count}"
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
        self.log = Logger(self.options.get('log_level', 2), f'Stats {self.pid}')
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

        register_gym_mods()
        env = gym.make(self.options.get('env_name'))

        while True:

            self.process_controller_messages(env)

            if self.tasks['stop']:
                print(F"stats collector {self.pid} stopping due to request from controller")
                return


class AsyncQLearningController:

    def __init__(self, options=None):
        self.options = Options(options)
        self.q_func = FunctionApprox(self.options)
        self.q_func.load_weights()
        self.workers = []
        self.stats_collector = None
        self.log = Logger(self.options.get('log_level', 2), "Controller")

    def message_all_workers(self, msg_code, content):
        msg = (msg_code, content)
        for worker, worker_queue in self.workers:
            # send message to each worker
            # print(f"sending : {msg_code}")
            try:
                worker_queue.put(msg)
            except Exception as e:
                print(f"Queue put failed in controller")
                print(e)
            # print(f"sent : {msg_code}")

    def message_stats_collector(self, msg_code, content):
        msg = (msg_code, content)
        stats_queue, worker = self.stats_collector
        try:
            # print(f"Send {msg_code} message to stats collector")
            stats_queue.put(msg)
        except Exception as e:
            print(f"stats_queue put failed in controller")
            print(e)

    def broadcast_value_network_weights(self):
        self.log.debug(f"broadcast network weights")
        value_network_weights = deepcopy(self.q_func.get_value_network_weights())
        self.log.debug(f"checksum value after call to set {self.q_func.weights_checksum(value_network_weights)}")
        self.message_all_workers('value_network_weights', value_network_weights)

    def update_value_network_weights(self, weight_deltas):
        self.log.debug(f"\nchecksum at start of update {self.q_func.get_value_network_checksum()}")
        value_network_weights = self.q_func.get_value_network_weights().copy()
        # print(f"value network weights before deltas {self.q_func.weights_checksum(value_network_weights)}")
        for name in value_network_weights:
            value_network_weights[name] += weight_deltas[name]

        # print(f"update value network weights {self.q_func.weights_checksum(value_network_weights)}")
        self.log.debug(f"checksum value before call to set {self.q_func.get_value_network_checksum()}")
        self.q_func.set_value_network_weights(value_network_weights)
        value_network_weights = deepcopy(self.q_func.get_value_network_weights())
        self.log.debug(f"checksum value after call to set {self.q_func.weights_checksum(value_network_weights)}")
        self.message_all_workers('value_network_weights', value_network_weights)

    def train(self):
        print(f"{os.getppid()}: Setting up workers to run asynch q-learning")
        # Need to make sure workers are spawned rather than forked otherwise keras gets into a deadlock. Which seems
        # to be due to multiple processes trying to share the same default tensorflow graph.
        mp.set_start_method('spawn')

        # TODO : does this need a max size setting?
        network_gradients_queue = mp.Queue()
        info_queue = mp.Queue()

        # set up the workers

        for _ in range(self.options.get('num_workers', 1)):
            worker_queue = mp.Queue()
            worker = AsyncQLearnerWorker(worker_queue, network_gradients_queue, info_queue, self.options)
            worker.daemon = True    # helps tidy up child processes if parent dies.
            worker.start()
            self.workers.append((worker, worker_queue))

        # Send the workers the starting weights
        starting_weights = deepcopy(self.q_func.get_value_network_weights())
        # print(f"init weights for workers {self.q_func.weights_checksum(starting_weights)}")
        self.message_all_workers('reset_network_weights', starting_weights)

        # set up stats_gatherer
        stats_queue = mp.Queue()
        worker = AsyncStatsCollector(stats_queue, self.options)
        worker.daemon = True    # helps tidy up child processes if parent dies.
        worker.start()
        self.stats_collector = (stats_queue, worker)

        total_steps = 0
        episode_count = 0
        target_sync_counter = self.options.get('target_net_sync_steps', 1)
        best_reward = None
        best_episode = 0
        grads_received = 0
        grads_received_errors = 0

        while episode_count < self.options.get('episodes', 1):

            gradient_messages = []
            try:
                while True:
                    gradient_messages.append(network_gradients_queue.get(False))
                    grads_received += 1
                    self.log.debug(f"got network gradients. Total received={grads_received}, "
                                   f"error count={grads_received_errors}")

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                grads_received_errors += 1
                self.log.error(f"Failed to get gradients off the queue. Total received={grads_received}, "
                               f"error count={grads_received_errors}", e)

            if len(gradient_messages) > 0:
                # print(f"apply the gradients, count of messages = {len(gradient_messages)}")
                # weight_deltas = None
                accum_grads = None
                self.log.debug(f"accumulate grads from {len(gradient_messages)} messages")
                for (steps, parameter_grads) in gradient_messages:
                    total_steps += steps
                    target_sync_counter -= steps
                    # get changes
                    if accum_grads is None:
                        accum_grads = parameter_grads
                    else:
                        accum_grads += parameter_grads

                self.log.debug(f"apply grads to network")
                self.q_func.apply_grads(parameter_grads)

                # Let workers know of the changes.
                self.log.debug(f"broadcast updated network weights")
                self.broadcast_value_network_weights()

                # if weight_deltas is not None:
                #     # if len(gradient_messages) > 1:
                #     #     # multiple gradient messages - need to take an average...
                #     #     # print(f"Multiple gradient messages ({len(gradient_messages)}), so take an average")
                #     #     # TODO: check how we should handle this
                #     #     for i in range(len(gradient_deltas)):
                #     #         gradient_deltas[i] /= len(gradient_messages)
                #      #        self.update_value_network_weights(weight_deltas)

            if target_sync_counter <= 0:
                # synch the target weights with the value weights, then let the workers know.
                self.q_func.synch_value_and_target_weights()
                network_weights = self.q_func.get_target_network_weights()
                # print(f"target network synch {self.q_func.weights_checksum(network_weights)}")
                self.message_all_workers('target_network_weights', network_weights)
                target_sync_counter = self.options.get('target_net_sync_steps', 1)

            try:
                # print(f"about to read from info queue")
                info = info_queue.get(False)
                episode_count += 1
                if best_reward is None or  best_reward < info['total_reward']:
                    best_reward = info['total_reward']
                    best_episode = episode_count
                print(f"Total steps {total_steps} : Episode {episode_count} took {info['steps']} steps"
                      f" : reward = {info['total_reward']} : pid = {info['worker']}"
                      f" : value_net = {info['value_network_checksum']:0.4f}"
                      f" : target_net = {info['target_network_checksum']:0.4f}"
                      f" : epsilon = {info['epsilon']:0.4f}"
                      f" : best_reward (episode) = {best_reward} ({best_episode})")

                if episode_count % self.options.get('stats_every') == 0:
                    self.q_func.save_weights()
                    weights = self.q_func.get_value_network_weights()
                    self.message_stats_collector('play', {'weights': weights, 'episode_count': episode_count})

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                print(f"Failed to read info_queue")
                print(e)

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
            print("Failed to close the workers cleanly")
            print(e)

def register_gym_mods():
    gym.envs.registration.register(
        id='MountainCarMyEasyVersion-v0',
        entry_point='gym.envs.classic_control.mountain_car:MountainCarEnv',
        max_episode_steps=250,      # MountainCar-v0 uses 200
        reward_threshold=-110.0
    )

def create_and_run_agent():

    timer = Timer()
    timer.start("agent")
    agent = AsyncQLearningController(options={
        # 'env_name': "Acrobot-v1",
        # 'observation_shape': (6, ),
        # 'actions': [0, 1, 2],
        # 'env_name': "MountainCar-v0",
        # 'env_name': "MountainCarMyEasyVersion-v0",
        # 'observation_shape': (2, ),
        # 'actions': [0, 1, 2],
        'env_name': "CartPole-v1",
        # 'render': "human",
        'observation_shape': (4, ),
        'actions': [0, 1],
        'num_workers': 2,
        'episodes': 500,
        'async_update_freq': 200,
        'target_net_sync_steps': 2000,
        'sync_tau': 1.0,
        'adam_learning_rate': 0.001,
        'discount_factor': 0.85,
        'load_weights': False,
        'save_weights': True,
        'stats_epsilon': 0.01,
        'epsilon': 0.75,
        'epsilon_min': 0.1,
        'epsilon_decay_episodes': 150,
        'stats_every': 10,  # how frequently to collect stats
        'play_avg': 1,      # number of games to average
        'log_level': 2,     # debug=1, info=2, warn=3, error=4
    })

    agent.train()
    timer.stop('agent')
    elapsed = timer.event_times['agent']
    hours = int(elapsed / 3600)
    elapsed = elapsed - 3600*hours
    mins = int(elapsed / 60)
    secs = int(elapsed - 60*mins)
    print(f"Total time taken by agent is {hours} hours {mins} mins {secs} secs")

if __name__ == '__main__':
    create_and_run_agent()