import gym
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque

env = gym.make("ALE/DemonAttack-v5", render_mode='human')
env.reset()

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

    def construct_av_network(self, num_actions: int):
        """Construct a neural network for producing actions in 
        the environment"""
        
        action_value_network = Sequential()

        action_value_network.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(8, 16, 1)))
        action_value_network.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        action_value_network.add(Flatten())
        action_value_network.add(Dense(256, activation='relu'))
        action_value_network.add(Dense(num_actions, activation='relu'))

        # compile the model
        optimiser = Adam(learning_rate=0.001)
        action_value_network.compile(optimizer=optimiser, loss="mean_squared_error", metrics=['accuracy'])

        return action_value_network

    def initialise_network_buffer(self, num_actions: int):
        """Create the networks and experience buffer for 
        a DQN agent"""

        # Initialise Experience Buffer D
        self.experience_buffer = ExperienceBuffer(self.max_buffer_len)

        self.q1 = self.construct_av_network(num_actions)
        # Create a fresh replica of the q1 model
        q1_copy = clone_model(self.q1)

        # Set weights of q2 to be the same as those of q1 to avoid
        # facing mismatch issues when using initialisation algorithms
        self.q2 = q1_copy.set_weights(self.q1.get_weights())

    def get_q1_a_star(self):
        """Retrieve best action to take in current state (a*)"""
        pass
    
    def epsilon_greedy_selection(self, num_actions: int, possible_actions: list):

        # Copy to avoid overwriting possible actions elsewhere
        valid_actions = possible_actions.copy()
        # Select A* or other action with epsilon-greedy policy
        a_star_chance = 1-self.epsilon + (self.epsilon/num_actions)
        lower_valued_chance = self.epsilon/num_actions

        # Generate a float for which epsilon will function as a threshold
        generated_float = np.random.rand()

        # Calculate best action estimate (a_star)
        a_star = self.get_q1_a_star()

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
    experience_buffer_size = 500
    epsilon = 0.15
    gamma = 0.9


    # Initialise a new environment
    env = gym.make("ALE/DemonAttack-v5", render_mode='human')
    prev_state = env.reset()

    # Collect info about environment actions for constructing network outputs
    num_actions = env.action_space.n
    print(num_actions)
    possible_actions = [n for n in range(num_actions)]
    
    # Initialise a new DQNAgent
    agent = DQNAgent(epsilon=epsilon, max_buffer_len=experience_buffer_size, gamma=gamma)
    agent.initialise_network_buffer(num_actions)

    # Set terminal to False initially for looping
    terminal=False

    # Fill replay buffer sith initial random wandering
    for _ in range(experience_buffer_size):
        action = np.random.choice(possible_actions)
        next_state, reward, terminal, _, _ = env.step(action)
        # Add experience to the buffer
        agent.experience_buffer.add(prev_state, action, reward, next_state, terminal)

    # Reset emnvironment now that replay buffer filled
    env = gym.make("ALE/DemonAttack-v5", render_mode='human')
    prev_state = env.reset()
    terminal = False

    while not terminal:
        action = 2 # Get action from network
        break

main()


# Select action greedily - use best -> Otherwise use random action