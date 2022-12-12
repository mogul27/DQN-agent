import multiprocessing as mp
from queue import Empty
import os
from pathlib import Path
import time
import random

import gym
import cv2

from dqn_utils import Options, DataWithHistory, Timer

# import keras
from keras.models import Sequential
from keras import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.losses import Huber
from keras.optimizers import Adam

import numpy as np


class EGreedyPolicy:
    """ Assumes every state has the same possible actions.
    """

    def __init__(self, epsilon, q_func, possible_actions, epsilon_decay_span=None, epsilon_min=None):
        """ e-greedy policy based on the supplied q_table.

        :param epsilon: small epsilon for the e-greedy policy. This is the probability that we'll
                                    randomly select an action, rather than picking the best.
        :param q_func: Approximates q values for state action pairs so we can select the best action.
        :param possible_actions: actions to be selected from with epsilon probability
        :param epsilon_decay_span: the number of calls over which to decay epsilon
        :param epsilon_min: the min value epsilon can be after decay
        """
        self.q_func = q_func
        self.epsilon = epsilon
        self.possible_actions = possible_actions
        if epsilon_decay_span is None:
            self.epsilon_min = epsilon
            self.epsilon_decay = 0
        else:
            if epsilon_min is None:
                self.epsilon_min = 0
            else:
                self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_decay_span

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

        # decay epsilon for next time
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        return action

    def random_action(self):
        return random.choice(self.possible_actions)


class FunctionApprox:

    def __init__(self, actions, update_batch_size=5, adam_learning_rate=0.0001):
        self.actions = actions
        self.q_hat = self.build_cnn(adam_learning_rate)
        self.q_hat_target = self.build_cnn(adam_learning_rate)
        self.clone_weights()
        self.update_batch_size = update_batch_size
        self.batch = []

    def save_weights(self, file_name):
        self.q_hat.save_weights(file_name)

    def load_weights(self, file_name):
        if Path(file_name).exists():
            self.q_hat.load_weights(file_name)
            self.q_hat_target.load_weights(file_name)

    def clone_weights(self):
        # Copy the weights from action_value network to the target action_value network
        self.q_hat_target.set_weights(self.q_hat.get_weights())

    def get_value_network_weights(self):
        return self.q_hat.get_weights()

    def set_value_network_weights(self, weights):
        self.q_hat.set_weights(weights)

    def get_target_network_weights(self):
        self.q_hat_target.get_weights()

    def set_target_network_weights(self, weights):
        self.q_hat_target.set_weights(weights)

    def build_cnn(self, adam_learning_rate):
        # Crete CNN model to predict actions for states.

        try:
            # TODO : give all the layers and models names to indicate worker / controller ?
            # inputs = Input((84, 84, 4))
            # conv_1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu')(inputs)
            # conv_2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(conv_1)
            # conv_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv_2)
            # flatten = Flatten()(conv_3)
            # dense_1 = Dense(512, activation='relu')(flatten)
            # outputs = Dense(4, activation=None)(dense_1)
            # cnn = Model(inputs=inputs, outputs=outputs)
            cnn = Sequential()
            # TODO : The async paper uses fewer and smaller.
            cnn.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
            cnn.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
            cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
            cnn.add(Flatten())
            cnn.add(Dense(512, activation='relu'))
            cnn.add(Dense(4, activation=None))
            cnn.summary()

            # compile the model
            cnn.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=adam_learning_rate))

            return cnn
        except Exception as e:
            print(f"failed to create model : {e}")

    def transpose_states(self, states):
        # states is a 4D array (N, X,Y,Z) with
        # N = number of states,
        # X = state and history, for CNN we need to transpose it to (N, Y,Z,X)
        # and also add another level.
        return np.transpose(np.array(states), (0, 2, 3, 1))

    def get_value(self, state, action):
        prediction = self.q_hat.predict_on_batch(self.transpose_states([state]))
        return prediction[0][action]

    def get_all_action_values(self, states):
        return self.q_hat.predict_on_batch(self.transpose_states(states))

    def get_target_value(self, state, action):
        prediction = self.q_hat_target.predict_on_batch(self.transpose_states([state]))
        return prediction[0][action]

    def get_max_target_value(self, state):
        prediction = self.q_hat_target.predict_on_batch(self.transpose_states([state]))
        return max(prediction[0])

    def get_max_target_values(self, states):
        predictions = self.q_hat_target.predict_on_batch(self.transpose_states(states))
        return predictions.max(axis=1)

    def best_action_for(self, state):
        prediction = self.q_hat.predict_on_batch(self.transpose_states([state]))
        return np.argmax(prediction[0])

    def update_batch(self, batch):
        # do the update in batches
        states = np.array([s for (s, new_action_value) in batch])
        states = self.transpose_states(states)
        new_action_values = np.array([new_action_value for (s, new_action_value) in batch])

        return self.q_hat.train_on_batch(states, new_action_values)

    def update(self, state, new_action_values):
        # do the update in batches
        self.batch.append((state, new_action_values))
        if len(self.batch) < self.update_batch_size:
            return

        states = np.array([s for (s, new_action_value) in self.batch])
        states = self.transpose_states(states)
        new_action_values = np.array([new_action_value for (s, new_action_value) in self.batch])

        self.q_hat.train_on_batch(states, new_action_values)

        # clear the batch.
        self.batch = []


class AsyncQLearnerWorker(mp.Process):

    def __init__(self, messages, network_gradient_queue, info_queue, options=None):
        super().__init__()
        # messages is a mp.Queue used to receive messages from the controller
        self.messages = messages
        self.network_gradient_queue = network_gradient_queue
        self.info_queue = info_queue
        self.options = Options(options)

        self.options.default('async_update_freq', 5)
        self.options.default('env_name', "ALE/Breakout-v5")
        self.options.default('actions', [0, 1, 2, 3])

        # set these up at the start of run. Saves them being pickled/reloaded when the new process starts.
        self.tasks = None
        self.q_func = None
        self.policy = None
        self.gradient_deltas = None
        self.initialised = False
        self.num_lives = 0
        self.discount_factor = 0.99

    def get_latest_tasks(self):
        """ Get the latest tasks from the controller.

        Overrides any existing task of the same name.
        """

        read_messages = True
        while read_messages:
            try:
                msg_code, content = self.messages.get(False)
                # print(f"worker {self.pid}: got message {msg_code}")
                if msg_code in self.tasks:
                    self.tasks[msg_code] = content
                else:
                    print(f"worker {self.pid}: ignored unrecognised message {msg_code}")
            except Empty:
                # Nothing to process, so just carry on
                read_messages = False
            except Exception as e:
                print(f"read messages failed")
                print(e)

    def process_controller_messages(self):
        """ see if there are any messages from the controller, and process accordingly
        """
        self.get_latest_tasks()
        if self.tasks['stop']:
            return

        if self.tasks['value_network_weights'] is not None:
            # print(f"worker {self.pid}: update the value network weights")
            self.q_func.set_value_network_weights(self.tasks['value_network_weights'])
            self.tasks['value_network_weights'] = None

        if self.tasks['target_network_weights'] is not None:
            # print(f"worker {self.pid}: update the target network weights")
            self.q_func.set_target_network_weights(self.tasks['target_network_weights'])
            self.tasks['target_network_weights'] = None

        if self.tasks['reset_network_weights'] is not None:
            # print(f"worker {self.pid}: reset both value and target network weights")
            network_weights = self.tasks['reset_network_weights']
            self.q_func.set_value_network_weights(network_weights)
            self.q_func.set_target_network_weights(network_weights)

            self.gradient_deltas = network_weights
            for layer_weights in network_weights:
                layer_weights.fill(0.1)
            self.initialised = True
            self.tasks['reset_network_weights'] = None

    def process_batch(self, state_action_batch):
        """ Process the batch and update the q_func, value function approximation.

        :param state_action_batch: List of tuples with state, action, reward, next_state, terminated
        """
        # process the batch - get the values and target values in single calls
        # TODO : make this neater
        states = []
        next_states = []
        for data_item in state_action_batch:
            states.append(data_item[0])
            next_states.append(data_item[3])
        states = np.array(states)
        next_states = np.array(next_states)

        qsa_action_values = self.q_func.get_all_action_values(states)
        next_state_action_values = self.q_func.get_max_target_values(next_states)
        discounted_next_qsa_values = self.discount_factor * next_state_action_values
        updates = []

        for data_item, qsa_action_value, discounted_next_qsa_value in zip(state_action_batch, qsa_action_values, discounted_next_qsa_values):
            s, a, r, next_s, terminated = data_item
            if terminated:
                y = r
            else:
                y = r + discounted_next_qsa_value

            delta = qsa_action_value[a] - y

            # update the action value to move closer to the target
            qsa_action_value[a] = y

            updates.append((s, qsa_action_value))

        losses = self.q_func.update_batch(updates)
        return losses

    def reformat_observation(self, obs, previous_obs=None):
        # take the max from obs and last_obs to reduce odd/even flicker that Atari 2600 has
        if previous_obs is not None:
            np.maximum(obs, previous_obs, out=obs)
        # reduce merged greyscalegreyscale from 210,160 down to 84,84
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

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
        print(f"I am a worker, my PID is {self.pid}")

        # keep track of requests from the controller by recording them in the tasks dict.
        self.tasks = {
            'value_network_weights': None,
            'target_network_weights': None,
            'reset_network_weights': None,
            'stop': None
        }

        self.q_func = FunctionApprox(self.options.get('actions'))

        # TODO : add epsilon to options
        self.policy = EGreedyPolicy(1.0, self.q_func, self.options.get('actions'),
                                    epsilon_decay_span=10000, epsilon_min=0.1)

        self.gradient_deltas = None
        self.initialised = False

        steps = 0
        steps_since_async_update = 0

        env = gym.make(self.options.get('env_name'), obs_type="grayscale", frameskip=1)
        terminated = True
        state_and_history = None
        action = -1
        total_undiscounted_reward = 0
        state_action_buffer = []

        while True:

            self.process_controller_messages()

            if self.tasks['stop']:
                print(F"Worker {self.pid} stopping due to request from controller")
                return

            if len(state_action_buffer) >= self.options.get('async_update_freq'):
                # We've processed enough steps to do an update.
                weights_before = self.q_func.get_value_network_weights()
                losses = self.process_batch(state_action_buffer)
                weights_after = self.q_func.get_value_network_weights()
                weight_deltas = [w_after - w_before for w_before, w_after in zip(weights_before, weights_after)]
                # send the gradient deltas back to the controller
                try:
                    # print(f"worker about to send weights deltas to controller. steps {steps}")
                    self.network_gradient_queue.put((len(state_action_buffer), weight_deltas))
                except Exception as e:
                    print(f"worker failed to send weight deltas")
                    print(e)
                state_action_buffer = []

            if self.initialised:
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

                    state_action_buffer.append((np.array(state), action, reward, np.array(next_state),
                                                terminated or life_lost))

                    steps += 1
                    steps_since_async_update += 1

                    total_undiscounted_reward += reward
                    if terminated:
                        try:
                            # print(f"send info message from worker")
                            # print(f"worker about to send weights deltas to controller")
                            self.info_queue.put({'total_reward': total_undiscounted_reward, 'steps': steps})
                        except Exception as e:
                            print(f"worker failed to send weight deltas")
                            print(e)
                        # print(f"Worker {self.pid}: Finished episode after {steps} steps. "
                        #       f"Total reward {total_undiscounted_reward}")
                        steps = 0

                    state, action = next_state, next_action


class AsyncQLearningController:

    def __init__(self, actions, options=None):
        self.actions = actions
        self.options = Options(options)
        self.options.default('num_workers', 1)
        self.options.default('total_steps', 500)
        self.q_func = FunctionApprox(actions, options)
        self.workers = []

    def message_all_workers(self, msg_code, content):
        msg = (msg_code, content)
        for worker, worker_queue in self.workers:
            # send message to each worker
            # print(f"controller sending : {msg_code}")
            try:
                worker_queue.put(msg)
            except Exception as e:
                print(f"Queue put failed in controller")
                print(e)
            # print(f"controller sent : {msg_code}")

    def update_value_network_weights(self, gradient_deltas):
        value_network_weights = self.q_func.get_value_network_weights()
        for (weights, deltas) in zip(value_network_weights, gradient_deltas):
            weights += deltas

        self.q_func.set_value_network_weights(value_network_weights)

        self.message_all_workers('value_network_weights', value_network_weights)

    def train(self):
        print(f"Controller {os.getppid()}: Setting up workers to run asynch q-learning")
        # Need to make sure workers are spawned rather than forked otherwise keras gets into a deadlock. Which seems
        # to be due to multiple processes trying to share the same default tensorflow graph.
        mp.set_start_method('spawn')

        # TODO : does this need a max size setting?
        network_gradients_queue = mp.Queue()
        info_queue = mp.Queue()

        # set up the workers

        for _ in range(self.options.get('num_workers')):
            worker_queue = mp.Queue()
            worker = AsyncQLearnerWorker(worker_queue, network_gradients_queue, info_queue, self.options)
            worker.daemon = True    # helps tidy up child processes if parent dies.
            worker.start()
            self.workers.append((worker, worker_queue))

        # Send the workers the starting weights
        self.message_all_workers('reset_network_weights', self.q_func.get_value_network_weights())
        total_steps = 0
        episode_count = 0
        target_sync_counter = self.options.get('target_net_sync_steps')

        while total_steps < self.options.get('total_steps'):

            gradient_messages = []
            try:
                while True:
                    # print(f"controller about to read from network gradients queue")
                    gradient_messages.append(network_gradients_queue.get(False))
                    # print(f"controller got network gradients")

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                print(f"Failed to get gradients off the queue")
                print(e)

            if len(gradient_messages) > 0:
                # print(f"apply the gradients, count of messages = {len(gradient_messages)}")
                gradient_deltas = None
                for (steps, deltas) in gradient_messages:
                    total_steps += steps
                    target_sync_counter -= steps
                    # update the value network weights.
                    if gradient_deltas is None:
                        gradient_deltas = deltas
                    else:
                        gradient_deltas += deltas

                if gradient_deltas is not None:
                    self.update_value_network_weights(gradient_deltas)

            if target_sync_counter <= 0:
                self.message_all_workers('target_network_weights', self.q_func.get_target_network_weights())
                target_sync_counter = self.options.get('target_net_sync_steps')

            try:
                # print(f"controller about to read from info queue")
                info = info_queue.get(False)
                episode_count += 1
                print(f"Total steps {total_steps} : Episode {episode_count} took {info['steps']} steps "
                      f": reward = {info['total_reward']}")

            except Empty:
                # Nothing to process, so just carry on
                pass

            except Exception as e:
                print(f"Failed to read info_queue")
                print(e)

        # close down the workers
        print(f"All done, {total_steps} steps processed - close down the workers")
        self.message_all_workers('stop', True)
        time.sleep(1)   # give them a chance to stop
        try:
            for worker, _ in self.workers:
                worker.terminate()
                worker.join(1)
        except Exception as e:
            print("Failed to close the workers cleanly")
            print(e)


def create_and_run_agent():
    agent = AsyncQLearningController([0, 1, 2, 3], options={
        'num_workers': 4,
        'total_steps': 100000,
        'async_update_freq': 5,
        'target_net_sync_steps': 250
    })
    agent.train()


if __name__ == '__main__':
    create_and_run_agent()