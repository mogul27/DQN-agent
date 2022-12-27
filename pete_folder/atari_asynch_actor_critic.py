import multiprocessing as mp
from queue import Empty
import os
from pathlib import Path
import time
import random
import math
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


class ACPolicy:
    """ Assumes every state has the same possible actions.
    """

    def __init__(self, actor_critic_network, options=None):
        """ policy using the supplied actor-critic network

        :param actor_critic_network: The actor critic network to predict actions and state values
        :param options: can contain:
            'epsilon': small epsilon for the e-greedy policy. This is the probability that we'll
                       randomly select an action, rather than picking the best.
            'epsilon_decay_episodes': the number of episodes over which to decay epsilon
            'epsilon_min': the min value epsilon can be after decay
        """
        self.options = Options(options)
        self.options.default('epsilon', 0.1)

        self.actor_critic_network = actor_critic_network
        self.epsilon = self.options.get('epsilon')
        self.possible_actions = self.options.get('actions')
        decay_episodes = self.options.get('epsilon_decay_episodes', 0)
        if decay_episodes == 0:
            self.epsilon_min = self.epsilon
            self.epsilon_decay = 0
        else:
            self.epsilon_min = self.options.get('epsilon_min', 0)
            # We can have multiple workers performing episodes, but each worker bases the decay on their own number
            # of episodes, not the overall number processed by all workers.
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / decay_episodes

    def get_actions_and_state_value(self, state):
        # Return softmax of actions, and the state value,
        action_probs, state_value = self.actor_critic_network(torch.Tensor(np.array([state])))
        # Squeeze the tensors to remove any dimensions of size 1
        action_probs = action_probs.squeeze()
        state_value = state_value.squeeze()
        return action_probs, state_value

    def select_action(self, state):
        """ The  policy selects the best action the majority of the time. However, a random action is chosen
        explore_probability amount of times.

        :param state: The sate to pick an action for
        :return: selected action
        """
        # Select an action for the state, use the best action most of the time.
        # However, with probability of explore_probability, select an action at random.
        # TODO for actor critic need to return the action and the state value, possibly also the probabilities too
        #      so, will need to make the call to q_func before deciding whether to take a random action. We need
        #      the state value.
        action_probs, _ = self.get_actions_and_state_value(state)

        if np.random.uniform() < self.epsilon:
            action = self.random_action()
        else:
            action = torch.argmax(action_probs).item()

        return action

    def sample_action(self, state):
        """ Selects an action from the distribution of actions returned by the actor.

        :return: (action, log action probability (logit), state value)
        """
        action_probs, state_value = self.get_actions_and_state_value(state)
        action_dist = torch.distributions.Categorical(probs=action_probs)

        action = action_dist.sample().item()

        action_logit = action_dist.logits[action]
        return action, action_logit, state_value

    def random_action(self):
        return random.choice(self.possible_actions)

    def decay_epsilon(self):
        # decay epsilon for next time - needs to be called at the end of an episode.
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)


# PyTorch models inherit from torch.nn.Module
class ActorCriticNetwork(nn.Module):
    """ Define neural network to be used by the DQN as the Actor-Critic.

    It has 2 output layers, one a softmax to give probability of taking an action, the other gives the state value.

    """
    def __init__(self, options):
        super().__init__()
        #   This is from the naure paper
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        # self.flatten1 = nn.Flatten()
        # self.fc1 = nn.Linear(3136, 512)
        # self.fc2 = nn.Linear(512, len(options.get('actions')))

        # This is from the asynch paper
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(2592, 256)
        # The action probabilities
        self.actor = nn.Linear(256, len(options.get('actions')))
        # The state value
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        # third layer not used in the asynch paper
        # x = torch.nn.functional.relu(self.conv3(x))
        x = self.flatten1(x)
        x = torch.nn.functional.relu(self.fc1(x))

        action_probabilities = torch.nn.functional.softmax(self.actor(x), dim=1)
        state_value = self.critic(x)

        return action_probabilities, state_value

    def checksum(self):
        chk = 0.0
        state_dict = self.state_dict()
        for name, layer_weights_or_bias in state_dict.items():
            chk += layer_weights_or_bias.sum()
        return chk


class AsynchActorCritic:

    def __init__(self, options, shared_actor_critic=None):
        self.options = Options(options)
        self.log = Logger(self.options.get('log_level', Logger.INFO), 'AsynchActorCritic')

        self.actions = self.options.get('actions')
        if shared_actor_critic is None:
            self.shared_actor_critic = self.build_neural_network()
        else:
            self.shared_actor_critic = shared_actor_critic
        self.optimizer = self.create_optimizer()

        self.local_actor_critic = self.build_neural_network()
        self.synch_shared_with_local()
        self.loss_fn = self.create_loss_function()

        self.discount_factor = self.options.get('discount_factor', 0.99)

        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        if self.work_dir is not None:
            # location for files.
            self.work_dir = Path(self.work_dir)
            if not self.work_dir.exists():
                self.work_dir.mkdir(parents=True, exist_ok=True)

        # Small value to stabilize division operations
        self.eps = np.finfo(np.float32).eps.item()

    def synch_shared_with_local(self):
        # Copy the weights from action_value network to the target action_value network
        sync_beta = self.options.get('sync_beta', 1.0)
        shared_state_dict = self.shared_actor_critic.state_dict().copy()
        local_state_dict = self.local_actor_critic.state_dict()
        for name in shared_state_dict:
            local_state_dict[name] = sync_beta * shared_state_dict[name] + (1.0 - sync_beta) * local_state_dict[name]
        self.set_local_weights(local_state_dict)

    def create_loss_function(self):
        return nn.HuberLoss()

    def create_optimizer(self):
        adam_learning_rate = self.options.get('adam_learning_rate', 0.0001)
        return torch.optim.Adam(self.shared_actor_critic.parameters(), lr=adam_learning_rate)

    def get_shared_weights(self):
        return self.shared_actor_critic.state_dict()

    def set_shared_weights(self, weights):
        self.shared_actor_critic.load_state_dict(weights)

    def get_shared_checksum(self):
        return self.weights_checksum(self.get_shared_weights())

    def get_local_weights(self):
        return self.local_actor_critic.state_dict()

    def set_local_weights(self, weights):
        self.local_actor_critic.load_state_dict(weights)

    def get_local_checksum(self):
        return self.weights_checksum(self.get_local_weights())

    def weights_checksum(self, state_dict):
        checksum = 0.0
        for name, layer_weights_or_bias in state_dict.items():
            checksum += layer_weights_or_bias.sum()
        return checksum

    def build_neural_network(self):
        # Create neural network model to predict actions for states.
        try:
            network = ActorCriticNetwork(self.options)

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
        # Not needed in pytorch
        return np.transpose(np.array(states), (0, 2, 3, 1))

    # TODO : these all need to change to handle the tuple return from the network
    def get_actions_and_state_value(self, state):
        # Return softmax of actions, and the state value,
        action_probs, state_value = self.local_actor_critic(torch.Tensor(np.array([state])))
        # Squeeze the tensors to remove any dimensions of size 1
        action_probs = action_probs.squeeze()
        state_value = state_value.squeeze()
        return action_probs, state_value

    def get_all_action_values(self, states):
        return self.local_actor_critic(torch.Tensor(states))

    def get_max_target_values(self, states):
        predictions = self.local_actor_critic(torch.Tensor(states))
        return torch.max(predictions, axis=1).values

    def best_action_for(self, state):
        state = torch.Tensor(np.array([state]))
        prediction = self.local_actor_critic(state)
        return torch.argmax(prediction).item()

    def expected_returns(self, rewards):
        returns = np.zeros(len(rewards), dtype=np.float32)
        i = len(rewards) - 1    # Start with the last one
        G = 0
        while i >= 0:
            G = rewards[i] + self.discount_factor * G
            returns[i] = G
            i -= 1

        # standardize the returns by
        returns = torch.Tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns

    # TODO : change this for actor critic
    def process_batch(self, states, actions, rewards, next_states, terminal, action_logits, state_values):
        """ Process the batch and update the actor-critic network.

        :param states: List of states
        :param actions: List of actions, one for each state
        :param rewards: List of rewards, one for each state
        :param next_states: List of next_states, one for each state
        :param terminal: List of terminal, one for each state
        :param action_logits: List of logits of the selected action, one for each state
        :param state_values: List of state values, one for each state
        """
        # If it's not
        not_terminal = torch.IntTensor([1 if not t else 0 for t in terminal])
        discounted_returns = self.expected_returns(rewards)

        # Zero the gradients for every batch!
        self.optimizer.zero_grad()

        # Compute the loss and its gradients.
        # state value for terminated state is zero
        # state_values = not_terminal * torch.stack(state_values)
        advantage = discounted_returns - torch.stack(state_values)
        actor_losses = -torch.stack(action_logits) * advantage
        actor_loss = actor_losses.sum()
        critic_losses = [self.loss_fn(v, r) for v, r in zip(state_values, discounted_returns)]
        critic_loss = torch.stack(critic_losses).sum()
        # # TODO : scale the losses?
        loss = actor_loss + critic_loss
        if loss == loss == math.inf:
            self.log.warn(f"loss is inf : {loss}")
        if loss == -math.inf:
            self.log.warn(f"loss is inf : {loss}")
        if loss != loss:
            # we have a NaN?
            self.log.warn(f"loss is nan : {loss}")
        self.log.debug(f"actions={actions}, actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}, loss={loss.item()}, ")

        loss.backward()
        # actor_loss.backward()
        # critic_loss.backward()

        # TODO : clip the gradient?

        # Adjust learning weights in the shared network
        for shared_param, local_param in zip(self.shared_actor_critic.parameters(), self.local_actor_critic.parameters()):
            shared_param._grad =local_param.grad

        self.optimizer.step()
        loss_value = loss.item(), actor_loss.item(), critic_loss.item()
        del loss
        del actor_loss
        del critic_loss
        return loss_value


class AsynchQLearnerWorker(mp.Process):

    def __init__(self, shared_actor_critic, messages, grad_update_queue, info_queue, options=None):
        super().__init__()
        self.shared_actor_critic = shared_actor_critic
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.grad_update_queue = grad_update_queue
        self.info_queue = info_queue
        self.options = Options(options)

        # set these up at the start of run. Saves them being pickled/reloaded when the new process starts.
        self.tasks = None
        self.asynch_actor_critic = None
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

    def asynch_grad_update(self, experiences):
        """ Process the batch of experiences, calling asynch_actor_critic to apply them.

        :param experiences: List of tuples with state, action, reward, next_state, terminated,
                                                action_logit, state_value
        """
        # swap list of tuples to individual lists
        states = []
        next_states = []
        actions = []
        rewards = []
        terminal = []
        action_logits = []
        state_values = []
        for s, a, r, ns, t, alp, sv in experiences:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            terminal.append(t)
            action_logits.append(alp)
            state_values.append(sv)

        loss = self.asynch_actor_critic.process_batch(states, actions, rewards, next_states, terminal,
                                                action_logits, state_values)

        try:
            self.log.trace(f"Sending number of steps to controller {len(experiences)}")
            self.grad_update_queue.put(len(experiences))
        except Exception as e:
            self.log.error(f"failed to send steps to grad_update_queue", e)

        del experiences[:]

        # Update local actor_critic with latest shared weights so that the policy uses the latest
        self.asynch_actor_critic.synch_shared_with_local()

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

        self.options.default('asynch_update_freq', 5)
        self.options.default('discount_factor', 0.99)

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'stop': None
        }

        self.asynch_actor_critic = AsynchActorCritic(self.options, self.shared_actor_critic)
        # Use a local copy of the actor_critic for the policy
        self.policy = ACPolicy(self.asynch_actor_critic.local_actor_critic, self.options)
        self.log.info(f"Created policy epsilon={self.policy.epsilon}, "
                      f"min={self.policy.epsilon_min}, decay={self.policy.epsilon_decay}")

        steps = 0
        steps_since_asynch_update = 0

        env = gym.make(self.options.get('env_name'), obs_type="grayscale")

        action = -1
        total_undiscounted_reward = 0
        experiences = []
        losses = []
        terminated = True   # default to true to get initial reset call
        life_lost = False

        while True:
            if self.worker_throttle is not None:
                time.sleep(self.worker_throttle)

            self.process_controller_messages()

            if self.tasks['stop']:
                self.log.info(F"stopping due to request from controller")
                return

            if len(experiences) >= self.options.get('asynch_update_freq', 5):
            # # if (terminated or life_lost) and len(experiences) > 0:
                loss = self.asynch_grad_update(experiences)
                losses.append(loss)
                experiences = []
                need_refresh = True

            if terminated:
                obs, info = env.reset()

                # State includes previous 3 frames.
                state = [DataWithHistory.empty_state() for i in range(3)]
                state.append(self.reformat_observation(obs))

                action, action_logit, state_value = self.policy.sample_action(state)

                if 'lives' in info:
                    # initialise the starting number of lives
                    self.num_lives = info['lives']

                total_undiscounted_reward = 0
                terminated = False
                need_refresh = False

            else:
                if need_refresh:
                    action, action_logit, state_value = self.policy.sample_action(state)
                    need_refresh = False

                # take a step and get the next action from the policy.

                # Take action A, observe R, S'
                obs, reward, terminated, truncated, info, life_lost = self.take_step(env, action)
                terminated = terminated or truncated    # treat truncated as terminated

                # take the last 3 frames from state as the history for next_state
                next_state = state[1:]
                next_state.append(obs)  # And add the latest obs

                # Choose A' from S' using policy derived from asynch_actor_critic
                # TODO : for actor critic we get the action probabilities and state value
                next_action, next_action_logit, next_state_value = self.policy.sample_action(next_state)
                # TODO : maybe skip adding every state and just add every nth?
                experiences.append((np.array(state), action, reward, np.array(next_state),
                                            terminated or life_lost, action_logit, state_value))

                steps += 1
                steps_since_asynch_update += 1

                total_undiscounted_reward += reward
                if terminated:
                    loss = self.asynch_grad_update(experiences)
                    losses.append(loss)
                    experiences = []
                    need_refresh = True
                    loss_total, actor_loss_total, critic_loss_total = 0, 0, 0
                    for loss, actor_loss, critic_loss in losses:
                        loss_total += loss
                        actor_loss_total += actor_loss
                        critic_loss_total += critic_loss
                    try:
                        # print(f"send info message from worker")
                        self.info_queue.put({
                            'total_reward': total_undiscounted_reward,
                            'steps': steps,
                            'worker': self.pid,
                            'epsilon': self.policy.epsilon,
                            'shared_network_checksum': self.asynch_actor_critic.get_shared_checksum(),
                            'local_network_checksum': self.asynch_actor_critic.get_local_checksum(),
                            'avg_loss': loss_total / len(losses),
                            'avg_actor_loss': actor_loss_total / len(losses),
                            'avg_critic_loss': critic_loss_total / len(losses)
                        })
                    except Exception as e:
                        self.log.error(f"worker failed to send to info_queue", e)
                    steps = 0
                    # apply epsilon decay after each episode
                    self.policy.decay_epsilon()

                state = next_state
                action = next_action
                action_logit = next_action_logit
                state_value = next_state_value


class AsynchStatsCollector(mp.Process):
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
        self.policy = None
        self.num_lives = 0
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
            # reset the weights for the policy's actor_critic_network to those passed from the controller
            self.policy.actor_critic_network.load_state_dict(contents['weights'])

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

        plot_filename = self.options.get('plot_filename', 'asynch_rewards.png')
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

        if total_reward > self.high_score:
            self.high_score = total_reward
            self.save_weights(episode_count, self.high_score)

        self.last_n_scores.append(total_reward)

        self.log.info(f"Play : Episode {episode_count}"
                      f" : Reward = {total_reward}"
                      f" : Action freq: {action_frequency}"
                      f" : Avg reward (last {len(self.last_n_scores)}) = {mean(self.last_n_scores):0.2f}")

        with open(self.info_file, 'a') as file:
            file.write(f'{episode_count}, {total_reward}, {self.policy.epsilon}, "{action_frequency}"\n')

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
            file.write(f"episode, reward, epsilon, value_net_checksum, action_frequency\n")

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'play': None,
            'stop': None
        }

        # TODO : just needs a simple thing to allow it to get the actions.
        actor_critic_network = ActorCriticNetwork(self.options)
        # use a policy with epsilon of 0.05 for playing to collect stats.
        self.policy = ACPolicy(actor_critic_network, self.options)
        self.policy.epsilon = self.options.get('stats_epsilon', 0.01)
        self.policy.min_epsilon = self.options.get('stats_epsilon', 0.01)
        print(f"stats collector {self.pid}: Created policy epsilon={self.policy.epsilon}, "
              f"min={self.policy.epsilon_min}, decay={self.policy.epsilon_decay}")

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
            torch.save(self.policy.actor_critic_network.state_dict(), weights_file)


class AsynchQLearningController:

    def __init__(self, options=None):
        self.options = Options(options)
        self.log = Logger(self.options.get('log_level', Logger.INFO), "Controller")

        # location for files.
        self.work_dir = self.options.get('work_dir', self.options.get('env_name'))
        self.work_dir = Path(self.work_dir)
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)

        # shared actor critic network
        self.shared_actor_critic = ActorCriticNetwork(self.options)
        self.load_weights()
        self.shared_actor_critic.share_memory()

        # other stuff
        self.workers = []
        self.stats_collector = None
        self.delta_fraction = self.options.get('delta_fraction', 1.0)
        self.details_file = self.create_details_file()

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
                torch.save(self.shared_actor_critic.state_dict(), weights_file)

    def load_weights(self):
        if self.options.get('load_weights', default=False):
            weights_file = self.get_weights_file()
            if weights_file is not None and weights_file.exists():
                state_dict = torch.load(weights_file)
                self.shared_actor_critic.load_state_dict(state_dict)

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

    def create_details_file(self):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        details_file = self.work_dir / f"episode-detail-{time_stamp}.csv"
        with open(details_file, 'a') as file:
            file.write(f"date-time,episode,total_steps,reward,pid,steps,avg_loss,avg_actor_loss,avg_critic_loss,"
                       f"shared_net,local_net,epsilon\n")
        return details_file

    def log_episode_detail(self, episode_detail, total_steps, episode):
        self.log.debug(f"Total steps {total_steps} : Episode {episode} took {episode_detail['steps']} steps"
                       f" : reward = {episode_detail['total_reward']} : pid = {episode_detail['worker']}"
                       f" : shared_net = {episode_detail['shared_network_checksum']:0.4f}"
                       f" : local_net = {episode_detail['local_network_checksum']:0.4f}"
                       f" : epsilon = {episode_detail['epsilon']:0.4f}")

        with open(self.details_file, 'a') as file:
            date_time = time.strftime('%Y/%m/%d-%H:%M:%S')
            file.write(f"{date_time},{episode},{total_steps}"
                       f",{episode_detail['total_reward']}"
                       f",{episode_detail['worker']}"
                       f",{episode_detail['steps']}"
                       f",{episode_detail['avg_loss']:0.6f}"
                       f",{episode_detail['avg_actor_loss']:0.6f}"
                       f",{episode_detail['avg_critic_loss']:0.6f}"
                       f",{episode_detail['shared_network_checksum']}"
                       f",{episode_detail['local_network_checksum']}"
                       f",{episode_detail['epsilon']}"
                       f"\n")

    def train(self):
        print(f"{os.getppid()}: Setting up workers to run asynch q-learning")

        grad_update_queue = mp.Queue()
        info_queue = mp.Queue()

        # set up the workers

        self.log.info(f"starting {self.options.get('num_workers', 1)} workers")
        # worker_eps = [0.5, 0.05, 0.15, 0.1, 0.2, 0.15, 0.25, 0.1]
        for i in range(self.options.get('num_workers', 1)):
            # epsilon = worker_eps[i % len(worker_eps)]
            # self.options.set('epsilon', epsilon)
            # self.options.set('epsilon_min', epsilon)
            worker_queue = mp.Queue()
            worker = AsynchQLearnerWorker(self.shared_actor_critic,
                                          worker_queue, grad_update_queue, info_queue, self.options)
            worker.daemon = True    # helps tidy up child processes if parent dies.
            worker.start()
            self.workers.append((worker, worker_queue))

        # set up stats_gatherer
        stats_queue = mp.Queue()
        worker = AsynchStatsCollector(stats_queue, self.options)
        worker.daemon = True    # helps tidy up child processes if parent dies.
        worker.start()
        self.stats_collector = (stats_queue, worker)

        total_steps = 0
        episode_count = self.options.get('start_episode_count', 0)
        last_episode = episode_count + self.options.get('episodes', 1)
        grad_updates = 0
        grads_update_errors = 0

        while episode_count < last_episode:

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

            try:
                self.log.trace(f"about to read from info queue")
                info = info_queue.get(False)
                episode_count += 1
                self.log_episode_detail(info, total_steps, episode_count)

                if episode_count % self.options.get('stats_every') == 0:
                    self.save_weights()
                    weights = self.shared_actor_critic.state_dict().copy()
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

        self.save_weights()

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

    agent = AsynchQLearningController(options)
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
        'work_dir': 'a3c/breakout',
        'env_name': "ALE/Breakout-v5",
        'actions': [0, 1, 2, 3],
        # 'work_dir': 'async/pong',
        # 'env_name': "ALE/Pong-v5",
        # 'actions': [0, 1, 2, 3, 4, 5],
        # 'render': "human",
        'actions': [0, 1, 2, 3],
        'num_workers': 8,
        'episodes': 5000,
        'asynch_update_freq': 1000,
        'sync_beta': 1.0,
        'adam_learning_rate': 0.0005,
        'discount_factor': 0.99,
        'load_weights': False,
        'save_weights': True,
        'stats_epsilon': 0.01,
        # 'epsilon': 1.0,       # using sample rather than e-greedy
        # 'epsilon_min': 0.1,
        # 'epsilon_decay_episodes': 500,
        'stats_every': 25,  # how frequently to collect stats
        'play_avg': 1,      # number of games to average
        'log_level': Logger.INFO,     # debug=1, info=2, warn=3, error=4
        'worker_throttle': 0.001,       # 0.001 a bit low for 2 workers, and 0.005 or 0.01 for 8
    }

    # for lr in [0.01, 0.001, 0.0001]:
    #     for num_workers in [1, 2, 4]:
    #         options['plot_filename'] = f'asynch_rewards_w_{num_workers}_lr_{lr}.png'
    #         options['plot_title'] = f"asynch Q-Learning {num_workers} workers, lr={lr}"
    #         options['num_workers'] = num_workers
    #         options['adam_learning_rate'] = lr
    #         create_and_run_agent(options)

    options['plot_filename'] = f'a3c_rewards_breakout.png'
    options['plot_title'] = f"Asynch Actor Critic Breakout {options['num_workers']} workers"
    # options['start_episode_count'] = 4000
    # options['epsilon'] = 0.1
    # options['num_workers'] = 1
    # options['episodes'] = 150
    # options['stats_every'] = 10
    # options['log_level'] = Logger.DEBUG

    create_and_run_agent(options)
