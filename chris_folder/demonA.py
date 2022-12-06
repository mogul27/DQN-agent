import gym
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

env = gym.make("ALE/DemonAttack-v5", render_mode='human')
env.reset()

class ExperienceBuffer:
    """ Maintain a memory of states and rewards from previous experience.

    Stores (state, action, reward, next_state) combinations encountered while playing the game up
    to a limit N
    Return random selection from the list
    """

    def __init__(self):
        self._data = deque()
        self.size = 0
        self.max_len = 1000

    def add(self, state, action, reward, next_state, terminal):
        if self.size >= self.max_len:
            # don't grow bigger, just lose one off the front.
            self.data.popleft()
        else:
            self.size += 1
        self._data.append((state, action, reward, next_state, terminal))

    def get_random_data(self, batch_size=1):
        # get a random batch of data
        #
        # :param batch_size: Number of data elements - default 1
        # :return: list of tuples of (state, action, reward, next_state)

        return random.choices(self._data, k=batch_size)


class NeuralNets:
    """Neural network class to contain the netowrks for actions and
    target network
    """

    def __init__(self, env):
        self.outputs = env.action_space
        self.num_actions = len(env.action_space)

    def construct_av_network(self):
        
        action_value_network = Sequential()

        action_value_network.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(8, 16, 1)))
        action_value_network.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        action_value_network.add(Flatten())
        action_value_network.add(Dense(256, activation='relu'))
        action_value_network.add(Dense(self.num_actions, activation='relu'))

        # compile the model
        optimiser = Adam(learning_rate=0.001)
        action_value_network.compile(optimizer=optimiser, loss="mean_squared_error", metrics=['accuracy'])

        return action_value_network

class DQNAgent:
    """Agent for Atari game DemoAttack using DQN"""
        
    self.q1 = construct_av_network()

    # Create a fresh replica of the q1 model
    q1_copy = clone_model(self.q1)

    self.q2 = q1_copy.set_weights(self.q1.get_weights())

