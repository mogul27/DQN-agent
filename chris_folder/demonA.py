import gym
import numpy as np
import random
from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque


class ExperienceBuffer:
    """ Maintain a memory of states and rewards from previous experience.

    Stores (state, action, reward, next_state) combinations encountered while playing the game up
    to a limit N
    Return random selection from the list
    """

    def __init__(self, max_buffer_len: int):
        self._data = deque()
        self.size = 0
        self.max_len = max_buffer_len

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


class DQNAgent:
    """Agent for Atari game DemoAttack using DQN"""

    def __init__(self, max_buffer_len, epsilon, gamma):
        """Initialise agent-specific attributes"""
        self.max_buffer_len = max_buffer_len
        self.epsilon = epsilon
        self.gamma=gamma

    def construct_av_network(self, num_actions: int, state_dims: tuple):
        """Construct a neural network for producing actions in 
        the environment"""

        action_value_network = Sequential()

        action_value_network.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=state_dims))
        action_value_network.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        action_value_network.add(Flatten())
        action_value_network.add(Dense(256, activation='relu'))
        action_value_network.add(Dense(num_actions, activation='relu'))

        # compile the model
        optimiser = Adam(learning_rate=0.001)
        action_value_network.compile(optimizer=optimiser, loss="mean_squared_error", metrics=['accuracy'])

        return action_value_network

    def initialise_network_buffer(self, num_actions: int, state_dims: tuple):
        """Create the networks and experience buffer for 
        a DQN agent"""

        # Initialise Experience Buffer D
        self.experience_buffer = ExperienceBuffer(self.max_buffer_len)

        self.q1 = self.construct_av_network(num_actions, state_dims)
        # Create a fresh replica of the q1 model
        q1_copy = clone_model(self.q1)

        # Set weights of q2 to be the same as those of q1 to avoid
        # facing mismatch issues when using initialisation algorithms
        self.q2 = q1_copy.set_weights(self.q1.get_weights())

    def get_q1_a_star(self, state):
        """Retrieve best action to take in current state (a*)"""
        q_vals= self.q1.predict(state)
        print(q_vals)
        a_star = np.argmax(q_vals)

        return a_star
    
    def epsilon_greedy_selection(self, num_actions: int, possible_actions: list, state):
        """Choose action A in state S using policy derived from q1(S,.,theta)"""

        # Copy to avoid overwriting possible actions elsewhere
        valid_actions = possible_actions.copy()
        # Select A* or other action with epsilon-greedy policy
        a_star_chance = 1-self.epsilon + (self.epsilon/num_actions)

        # Generate a float for which epsilon will function as a threshold
        generated_float = np.random.rand()

        # Calculate best action estimate (a_star)
        a_star = self.get_q1_a_star(state)

        if generated_float < a_star_chance:
            state_action = a_star
        else:
            # If not a_star action then remaining probabilities are equivalent to equiprobable
            # random choice between them
            valid_actions.remove(a_star)
            state_action = random.sample(valid_actions, 1)[0]

        return state_action

def main():

    # Set algorithm parameters
    experience_buffer_size = 1000000
    epsilon = 0.15
    gamma = 0.9


    # Initialise a new environment
    env = gym.make("ALE/DemonAttack-v5", render_mode='human', frameskip=1)
    # Apply preprocessing from original DQN paper including greyscale and cropping
    wrapped_env = gym.wrappers.AtariPreprocessing(env)
    prev_state = wrapped_env.reset()

    # Collect info about environment and actions for constructing network outputs
    num_actions = wrapped_env.action_space.n
    possible_actions = [n for n in range(num_actions)]
    state_dims = wrapped_env.observation_space.shape
    # Concatenate 1 to state dims to represent the number of channels
    # which is 1 because greyscale images used
    state_dims = state_dims + (1,)
    print("State Dims:", state_dims)
    print(type(state_dims))
    
    # Initialise a new DQNAgent
    agent = DQNAgent(epsilon=epsilon, max_buffer_len=experience_buffer_size, gamma=gamma)
    agent.initialise_network_buffer(num_actions, state_dims)

    # Set terminal to False initially for looping
    terminal=False

    # Fill replay buffer sith initial random wandering
    for _ in range(experience_buffer_size):
        action = np.random.choice(possible_actions)
        next_state, reward, terminal, _, _ = wrapped_env.step(action)
        # Add experience to the buffer
        agent.experience_buffer.add(prev_state, action, reward, next_state, terminal)

    # Reset emnvironment now that replay buffer filled
    env = gym.make("ALE/DemonAttack-v5", render_mode='human')
    wrapped_env = gym.wrappers.AtariPreprocessing(env)
    prev_state = wrapped_env.reset()
    terminal = False

    while not terminal:
        action = 2 # Get action from network
        break

main()


# Select action greedily - use best -> Otherwise use random action