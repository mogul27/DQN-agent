import multiprocessing as mp
from queue import Empty

from pathlib import Path
import time
import random
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt

import gym

from dqn_utils import Options, DataWithHistory, Timer, Logger, reformat_observation, take_step

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim

import numpy as np


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


class StatsPolicy:
    """ The supplied value_network returns Q values for each action available for a state.

    This policy simply selects the action that returns the largest state value.

    """

    def __init__(self, value_network):
        self.value_network = value_network

    def select_action(self, state):
        state = torch.Tensor(np.array([state]))
        prediction = self.value_network(state)
        return torch.argmax(prediction).item()

    def get_value_network_weights(self):
        return self.value_network.state_dict()

    def get_value_network_checksum(self):
        state_dict = self.get_value_network_weights()
        checksum = 0.0
        for name, layer_weights_or_bias in state_dict.items():
            checksum += layer_weights_or_bias.sum()
        return checksum


# PyTorch models inherit from torch.nn.Module
class QNetwork(nn.Module):
    """ Define neural network to be used as both q-network and target-network

    """
    def __init__(self, options):
        super().__init__()
        #   This is from the nature paper
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        # self.flatten1 = nn.Flatten()
        # self.fc1 = nn.Linear(3136, 512)
        # self.fc2 = nn.Linear(512, len(options.get('actions')))

        # This is from the async paper
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, len(options.get('actions')))

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        # third layer not used in the async paper
        # x = torch.nn.functional.relu(self.conv3(x))
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
        if self.options.get('lr_decay_episodes') is None or self.options.get('lr_decay_factor') is None:
            self.lr_sched = None
        else:
            self.lr_sched = torch.optim.lr_scheduler.StepLR(self.optimizer, self.options.get('lr_decay_episodes'),
                                                            self.options.get('lr_decay_factor') )

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
                    weights_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.q_hat.state_dict(), weights_file)

    def load_weights(self):
        if self.options.get('load_weights', default=False):
            weights_file = self.get_weights_file()
            if weights_file is not None and weights_file.exists():
                self.log.info(f"loading weights from {weights_file}")
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
        not_terminal = torch.IntTensor([1 if not t else 0 for t in terminal])

        # Make predictions for this batch
        predicted_action_values = self.get_all_action_values(np.array(states))
        next_state_action_values = self.get_max_target_values(np.array(next_states))
        discounted_rewards = np.zeros(len(rewards))
        if terminal[-1]:
            R = 0
        else:
            R = next_state_action_values[-1].item()

        i = len(discounted_rewards)
        while i > 0:
            i -= 1
            R = rewards[i] + self.discount_factor * R
            discounted_rewards[i] = R

        rewards = torch.Tensor(rewards)
        y_one_step = rewards + not_terminal * self.discount_factor * next_state_action_values
        y_n_step = torch.Tensor(discounted_rewards)
        new_action_values = predicted_action_values.clone()
        # new_action_values is a 2D tensor. Each row is the action values for a state.
        # actions contains a list with the action to be updated for each state (row in the new_action_values tensor)
        if self.options.get('n-step', False):
            new_action_values[torch.arange(new_action_values.size(0)), actions] = y_one_step
        else:
            new_action_values[torch.arange(new_action_values.size(0)), actions] = y_n_step

        # Zero the gradients for every batch!
        self.optimizer.zero_grad()

        # Compute the loss and its gradients
        losses = [self.loss_fn(v, r) for v, r in zip(predicted_action_values, new_action_values)]
        loss = torch.stack(losses).sum()
        # old_loss = self.loss_fn(predicted_action_values, new_action_values)
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()
        loss_value = loss.item()
        del predicted_action_values
        del next_state_action_values
        del losses
        del loss
        return loss_value

    def get_current_lr(self):
        if self.lr_sched is not None:
            return self.lr_sched.get_last_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']

    def end_episode(self):
        if self.lr_sched is not None:
            self.lr_sched.step()


class AsyncQLearnerWorker(mp.Process):

    def __init__(self, shared_value_network, shared_target_network, messages, grad_update_queue, info_queue,
                 options=None):
        super().__init__()
        self.shared_value_network = shared_value_network
        self.shared_target_network = shared_target_network
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.grad_update_queue = grad_update_queue
        self.info_queue = info_queue
        self.options = Options(options)

        # set these up at the start of run. Saves them being pickled/reloaded when the new process starts.
        self.tasks = None
        self.q_func_shared = None
        self.q_func_local = None
        self.policy = None
        self.initialised = False
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

        loss = self.q_func_shared.process_batch(states, actions, rewards, next_states, terminal)

        try:
            # self.log.trace(f"Sending number of steps to controller {len(experiences)}")
            self.grad_update_queue.put(len(experiences))
        except Exception as e:
            self.log.error(f"failed to send steps to grad_update_queue", e)

        del experiences[:]

        # Update local q_func with latest shared weights so that the policy uses the latest
        shared_weights = self.q_func_shared.get_value_network_weights().copy()
        self.q_func_local.set_value_network_weights(shared_weights)

        return loss


    def run(self):
        self.log = Logger(self.options.get('log_level', Logger.INFO), f"Worker {self.pid}")
        self.log.info(f"run started")

        self.options.default('async_update_freq', 5)
        self.options.default('discount_factor', 0.99)

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'stop': None
        }

        self.q_func_shared = FunctionApprox(self.options, self.shared_value_network, self.shared_target_network)
        # Use a local copy of the value network for the policy
        self.q_func_local = FunctionApprox(self.options)
        shared_weights = self.q_func_shared.get_value_network_weights().copy()
        self.q_func_local.set_value_network_weights(shared_weights)
        self.policy = EGreedyPolicy(self.q_func_local, self.options)
        self.log.info(f"Created policy epsilon={self.policy.epsilon}, "
                      f"min={self.policy.epsilon_min}, decay={self.policy.epsilon_decay}")

        steps = 0
        steps_since_async_update = 0
        num_lives = 0

        env = gym.make(self.options.get('env_name'), obs_type="grayscale", render_mode=self.options.get('render'))

        action = -1
        total_reward = 0
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
                state.append(reformat_observation(obs))

                action = self.policy.select_action(state)

                if 'lives' in info:
                    # initialise the starting number of lives
                    num_lives = info['lives']

                total_reward = 0
                terminated = False

            else:
                # take a step and get the next action from the policy.
                obs, reward, terminated, truncated, info, life_lost, num_lives = take_step(env, action, num_lives)
                terminated = terminated or truncated    # treat truncated as terminated

                # take the last 3 frames from state as the history for next_state
                next_state = state[1:]
                next_state.append(obs)  # And add the latest obs

                # Choose A' from S' using policy derived from q_func
                next_action = self.policy.select_action(next_state)
                # TODO : maybe skip adding every state and just add every nth?
                experiences.append((np.array(state), action, reward, np.array(next_state),
                                            terminated or life_lost))

                if terminated or life_lost:
                    loss = self.async_grad_update(experiences)
                    losses.append(loss)
                    experiences = []

                steps += 1
                steps_since_async_update += 1

                total_reward += reward
                if terminated:
                    try:
                        # print(f"send info message from worker")
                        self.info_queue.put({
                            'total_reward': total_reward,
                            'steps': steps,
                            'worker': self.pid,
                            'epsilon': self.policy.epsilon,
                            'value_network_checksum': self.q_func_shared.get_value_network_checksum(),
                            'target_network_checksum': self.q_func_shared.get_target_network_checksum(),
                            'avg_loss': sum(losses) / len(losses),
                            'lr': self.q_func_shared.get_current_lr()
                        })
                    except Exception as e:
                        self.log.error(f"worker failed to send to info_queue", e)
                    steps = 0
                    # apply epsilon decay after each episode
                    self.policy.decay_epsilon()
                    self.q_func_shared.end_episode()

                state, action = next_state, next_action


class AsyncStatsCollector(mp.Process):
    """ Gather some stats for the controller when asked to do so.

    Plays a game against the environment using the supplied weights.
    """

    def __init__(self, shared_value_network, messages, options=None):
        super().__init__()
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.options = Options(options)
        self.shared_value_network = shared_value_network
        self.value_network_copy = None

        # set these up at the start of run. Saves them being pickled/reloaded when the new process starts.
        self.tasks = None
        self.policy = None
        self.last_n_scores = None
        self.stats = None
        self.work_dir = None
        self.stats_file = None
        self.info_file = None
        self.log = None
        self.high_score = self.options.get('high_score', 0)

    def get_latest_tasks(self):
        """ Get the latest tasks from the controller.

        Overrides any existing task of the same name.
        """

        read_messages = True
        while read_messages:
            try:
                msg_code, content = self.messages.get(False)
                # self.log.trace(f"got message {msg_code}")
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
            self.log.debug(f"stats policy value network weights {self.policy.get_value_network_checksum()}")

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
        plots = []

        fig, ax = plt.subplots()

        ax.set_title(self.options.get('plot_title', 'Asynch Q-Learning'))
        ax.set_xlabel('episodes')
        ax.set_ylabel('reward')
        ax.grid(True)
        # ax1.set_ylim(0, 50)
        p, = ax.plot(x, rwd, label='rewards')
        plots.append(p)

        x = []
        avg_rwd = []
        last_few = []
        for i in range(len(self.stats)):
            stat = self.stats[i]
            x.append(stat[0])
            if len(last_few) >= 10:
                last_few.pop(0)
            last_few.append(stat[1])
            avg_rwd.append(sum(last_few)/len(last_few))
        p, = ax.plot(x, avg_rwd, label='moving average')
        plots.append(p)

        ax.legend(handles=plots)

        fig.tight_layout()

        plot_filename = self.options.get('plot_filename', 'async_rewards.png')
        plt.savefig(self.work_dir / plot_filename)
        plt.close('all')    # matplotlib holds onto all the figures if we don't close them.

    def play(self, env, episode_count):
        """ play a single episode using a greedy policy

        :param env: The env to use to run an episode
        :param episode_count: Number of episodes that have been run by the workers to generate the policy
        :return: The reward returned by using the policy against the env.
        """
        self.log.debug('stats_collector: play')
        shared_weights = self.shared_value_network.state_dict().copy()
        self.value_network_copy.load_state_dict(shared_weights)

        total_reward = 0
        num_lives = 5

        obs, info = env.reset()

        # State includes previous 3 frames.
        state = [DataWithHistory.empty_state() for i in range(3)]
        state.append(reformat_observation(obs))

        if 'lives' in info:
            # initialise the starting number of lives
            num_lives = info['lives']

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

            obs, reward, terminated, truncated, info, life_lost, num_lives = take_step(env, action, num_lives)
            total_reward += reward

            terminated = terminated or truncated    # treat truncated as terminated

            # remove the oldest frame from the state
            state.pop(0)
            state.append(obs)  # And add the latest obs

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        if total_reward > self.high_score:
            self.high_score = total_reward
            if self.options.get('save_stats_highscore_weights', default=False):
                self.save_weights(episode_count, self.high_score)

        self.last_n_scores.append(total_reward)

        self.log.info(f"Play : weights from episode {episode_count}"
                      f" : Reward = {total_reward}"
                      f" : value_net: {self.policy.get_value_network_checksum():0.4f}"
                      f" : Action freq: {action_frequency}"
                      f" : Avg reward (last {len(self.last_n_scores)}) = {mean(self.last_n_scores):0.2f}")

        with open(self.info_file, 'a') as file:
            file.write(f'{episode_count}, {total_reward}, "{action_frequency}"\n')

        return total_reward

    def run(self):
        self.log = Logger(self.options.get('log_level', Logger.INFO), f'Stats {self.pid}')
        self.log.info(f"started run")

        self.last_n_scores = deque(maxlen=10)
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
            file.write(f"episode,reward,action_frequency\n")

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'play': None,
            'stop': None
        }

        self.value_network_copy = QNetwork(self.options)
        self.policy = StatsPolicy(self.value_network_copy)
        self.log.info('created policy')

        env = gym.make(self.options.get('env_name'), obs_type="grayscale")

        while True:

            self.process_controller_messages(env)

            if self.tasks['stop']:
                print(F"stats collector {self.pid} stopping due to request from controller")
                return

    def get_weights_file(self, episode, high_score):
        if self.work_dir is not None:
            weights_file_name = self.options.get('weights_file', f"{self.options.get('env_name')}"
                                                                 f" s-{int(high_score)} e-{episode}.pth")
            weights_file_name = 'score_weights/' + weights_file_name
            return self.work_dir / weights_file_name

        return None

    def save_weights(self, episode, high_score):
        weights_file = self.get_weights_file(episode, high_score)
        if weights_file is not None:
            if not weights_file.parent.exists():
                weights_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy.get_value_network_weights(), weights_file)


class AsyncQLearningController:

    def __init__(self, options=None):
        self.options = Options(options)
        self.q_func = FunctionApprox(self.options)
        self.q_func.load_weights()
        self.shared_value_network = self.q_func.q_hat
        self.shared_target_network = self.q_func.q_hat_target
        self.shared_value_network.share_memory()
        self.shared_target_network.share_memory()
        self.workers = []
        self.stats_collector = None
        self.delta_fraction = self.options.get('delta_fraction', 1.0)
        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        # location for files.
        self.work_dir = Path(self.work_dir)
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        self.details_file = self.create_details_file()
        self.log = Logger(self.options.get('log_level', Logger.INFO), "Controller")

    def message_all_workers(self, msg_code, content):
        msg = (msg_code, content)
        for worker, worker_queue in self.workers:
            # send message to each worker
            # self.log.trace(f"sending : {msg_code}")
            try:
                worker_queue.put(msg)
            except Exception as e:
                self.log.error(f"Queue put failed in controller", e)
            # self.log.trace(f"sent : {msg_code}")

    def message_stats_collector(self, msg_code, content):
        msg = (msg_code, content)
        stats_queue, worker = self.stats_collector
        try:
            # self.log.trace(f"Send {msg_code} message to stats collector")
            stats_queue.put(msg)
        except Exception as e:
            self.log.error(f"stats_queue put failed in controller", e)

    def create_details_file(self):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        details_file = self.work_dir / f"episode-detail-{time_stamp}.csv"
        with open(details_file, 'a') as file:
            file.write(f"date-time,episode,total_steps,reward,pid,steps,avg_loss,value_net,target_net,epsilon,lr\n")
        return details_file

    def log_episode_detail(self, episode_detail, total_steps, episode):
        my_value_net = self.q_func.get_value_network_checksum()
        my_target_net = self.q_func.get_target_network_checksum()
        self.log.debug(f"Total steps {total_steps} : Episode {episode} took {episode_detail['steps']} steps"
                       f" : reward = {episode_detail['total_reward']} : pid = {episode_detail['worker']}"
                       f" : value_net = {episode_detail['value_network_checksum']:0.4f}"
                       f" : controller value_net = {my_value_net:0.4f}"
                       f" : target_net = {episode_detail['target_network_checksum']:0.4f}"
                       f" : controller target_net = {my_target_net:0.4f}"
                       f" : epsilon = {episode_detail['epsilon']:0.4f}"
                       f" : lr = {episode_detail['lr']:0.4f}"
                       )

        with open(self.details_file, 'a') as file:
            date_time = time.strftime('%Y/%m/%d-%H:%M:%S')
            file.write(f"{date_time},{episode},{total_steps}"
                       f",{episode_detail['total_reward']}"
                       f",{episode_detail['worker']}"
                       f",{episode_detail['steps']}"
                       f",{episode_detail['avg_loss']:0.6f}"
                       f",{episode_detail['value_network_checksum']}"
                       f",{episode_detail['target_network_checksum']}"
                       f",{episode_detail['epsilon']}"
                       f",{episode_detail['lr']}"
                       f"\n")

    def train(self):
        self.log.info(f"train : Setting up workers to run asynch q-learning")

        grad_update_queue = mp.Queue()
        info_queue = mp.Queue()

        # set up the workers

        self.log.info(f"starting {self.options.get('num_workers', 1)} workers")
        max_worker_eps = self.options.get('epsilon_maxs', [0.5])
        min_worker_eps = self.options.get('epsilon_mins', [0.1])
        for i in range(self.options.get('num_workers', 1)):
            epsilon_max = max_worker_eps[i % len(max_worker_eps)]
            epsilon_min = min_worker_eps[i % len(min_worker_eps)]
            worker_options = Options(self.options)
            worker_options.set('epsilon', epsilon_max)
            worker_options.set('epsilon_min', min(epsilon_min, epsilon_max))
            worker_queue = mp.Queue()
            worker = AsyncQLearnerWorker(self.shared_value_network, self.shared_target_network,
                                         worker_queue, grad_update_queue, info_queue, worker_options)
            worker.daemon = True    # helps tidy up child processes if parent dies.
            worker.start()
            self.workers.append((worker, worker_queue))

        # set up stats_gatherer
        stats_queue = mp.Queue()
        worker = AsyncStatsCollector(self.shared_value_network, stats_queue, self.options)
        worker.daemon = True    # helps tidy up child processes if parent dies.
        worker.start()
        self.stats_collector = (stats_queue, worker)

        total_steps = 0
        episode_count = self.options.get('start_episode_count', 0)
        last_episode = episode_count + self.options.get('episodes', 1)
        target_sync_counter = self.options.get('target_net_sync_steps', 1)
        grad_updates = 0
        grads_update_errors = 0

        while episode_count < last_episode:

            grad_update_messages = []
            try:
                while True:
                    grad_update_messages.append(grad_update_queue.get(False))
                    grad_updates += 1
                    # self.log.trace(f"got network gradients. Total received={grad_updates}, "
                    #                f"error count={grads_update_errors}")

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                grads_update_errors += 1
                self.log.error(f"Failed to get gradients off the queue. Total received={grad_updates}, "
                               f"error count={grads_update_errors}", e)

            if len(grad_update_messages) > 0:
                # self.log.trace(f"accumulate steps from {len(grad_update_messages)} messages")
                for steps in grad_update_messages:
                    total_steps += steps
                    target_sync_counter -= steps

            if target_sync_counter <= 0:
                # synch the target weights with the value weights, then let the workers know.
                self.q_func.synch_value_and_target_weights()
                target_sync_counter = self.options.get('target_net_sync_steps', 1)

            try:
                # self.log.trace(f"about to read from info queue")
                info = info_queue.get(False)
                episode_count += 1

                if episode_count % self.options.get('stats_every') == 0:
                    self.q_func.save_weights()
                    self.message_stats_collector('play', {'episode_count': episode_count})

                self.log_episode_detail(info, total_steps, episode_count)

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
        'work_dir': 'async/breakout',               # Location for output files
        'env_name': "ALE/Breakout-v5",
        # 'env_name': "BreakoutNoFrameskip-v4",
        'actions': [0, 1, 2, 3],
        # 'work_dir': 'async/pong',
        # 'env_name': "ALE/Pong-v5",                # Can easily train with other Atari games.
        # 'actions': [0, 1, 2, 3, 4, 5],
        # 'render': "human",
        'num_workers': 8,
        'worker_throttle': 0.001,                   # 0.001 seemed OK for 8 workers
        'episodes': 5000,
        'async_update_freq': 8,
        'target_net_sync_steps': 10000,
        'sync_beta': 1.0,
        'adam_learning_rate': 0.0001,
        'discount_factor': 0.99,
        # 'weights_file': "ALE/Breakout-v5.pth",    # defaults to env_name.pth, file saved to work_dir
        'load_weights': False,
        'save_weights': False,                      # If True, weights are saved ever 'stats_every' episodes
        # 'save_stats_highscore_weights': False,    # If True, stats collector saves weights from high scores
        # Epsilon for worker policies - each worker can have different epsilon, they're picked off the list
        'epsilon_maxs': [0.2, 0.1, 0.1, 0.1, 0.05, 0.02, 0.01, 0.0],
        'epsilon_mins': [0.2, 0.1, 0.1, 0.1, 0.05, 0.02, 0.01, 0.0],
        # Can reduce the epsilon later in training.
        # 'epsilon_maxs': [0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0],
        # 'epsilon_mins': [0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0],
        'epsilon_decay_episodes': 0,                # optional decay period for epsilon
        'stats_every': 25,                          # how frequently to collect stats
        'play_avg': 1,                              # number of games to average
        'log_level': Logger.INFO,                   # debug=1, info=2, warn=3, error=4
        'n_step':   True,                           # True for n_step Q-Learning, False for 1 step.
        # 'lr_decay_episodes': 100,
        # 'lr_decay_factor': 0.8,
    }

    # options['plot_filename'] = f'async_rewards_breakout.png'
    # options['plot_title'] = f"Asynch Q-Learning Breakout {options['num_workers']} workers"
    # options['start_episode_count'] = 20000

    # Just a basic test, useful to check it works, and debugging.
    options['num_workers'] = 1
    options['episodes'] = 10
    options['stats_every'] = 3
    # options['log_level'] = Logger.DEBUG

    create_and_run_agent(options)
