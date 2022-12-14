import numpy as np
import random
from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import RMSprop
from keras.losses import Huber
from dqn_utilities import ExperienceBuffer

class DQNAgent:
    """Agent for Atari game DemoAttack using DQN"""

    def __init__(self, max_buffer_len, epsilon, gamma):
        """Initialise agent-specific attributes"""
        self.max_buffer_len = max_buffer_len
        self.epsilon = epsilon
        self.gamma=gamma
        self.q1 = None
        self.q2 = None

    def construct_av_network(self, num_actions: int, state_dims: tuple):
        """Construct a neural network for producing actions in 
        the environment"""
        
        action_value_network = Sequential()
        action_value_network.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=state_dims))
        action_value_network.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        action_value_network.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        action_value_network.add(Flatten())
        action_value_network.add(Dense(512, activation='relu'))
        action_value_network.add(Dense(num_actions))

        # compile the model
        optimiser = RMSprop(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
        action_value_network.compile(optimizer=optimiser, loss=Huber(delta=1.0))

        return action_value_network

    def initialise_network_buffer(self, num_actions: int, state_dims: tuple):
        """Create the networks and experience buffer for 
        a DQN agent"""

        # Initialise Experience Buffer D
        self.experience_buffer = ExperienceBuffer(self.max_buffer_len)

        self.q1 = self.construct_av_network(num_actions, state_dims)

        # Create a fresh replica of the q1 model
        self.q2 = clone_model(self.q1)
        self.q2.set_weights(self.q1.get_weights())

        # Set weights of q2 to be the same as those of q1 to avoid
        # facing mismatch issues when using initialisation algorithms
        # self.q2 = q1_copy.set_weights(self.q1.get_weights())

    def get_q1_a_star(self, network_input):
        """Retrieve best action to take in current state (a*)"""

        q_vals = self.q1.predict_on_batch(network_input)[0]
        a_star = np.argmax(q_vals)

        return a_star

    def get_q2_preds(self, network_input, steps):
        """Retrieve the best action to take in given state"""
        q_vals = self.q2.predict_on_batch(network_input)
        best_action = np.argmax(q_vals[0]) 
        best_action_val = q_vals[0][best_action]

        return best_action_val

    def get_q1_action_values(self, network_input):
        q_vals = self.q1.predict_on_batch(network_input)[0]
        return q_vals
    
    def epsilon_greedy_selection(self, num_actions: int, possible_actions: list,
                                 network_input: np.ndarray):
        """Choose action A in state S using policy derived from q1(S,.,theta)"""

        # Copy to avoid overwriting possible actions elsewhere
        valid_actions = possible_actions.copy()
        # Select A* or other action with epsilon-greedy policy
        a_star_chance = 1-self.epsilon + (self.epsilon/num_actions)

        # Generate a float for which epsilon will function as a threshold
        generated_float = np.random.rand()

        # Calculate best action estimate (a_star)
        a_star = self.get_q1_a_star(network_input)

        if generated_float < a_star_chance:
            state_action = a_star
        else:
            # If not a_star action then remaining probabilities are equivalent to equiprobable
            # random choice between them
            valid_actions.remove(a_star)
            state_action = random.sample(valid_actions, 1)[0]

        return state_action

    def adjust_reward_lives(self, reward, info, prev_lives):

        lives = info['lives']

        if lives < prev_lives:
            reward = reward - 10

        return reward, lives
    
    def save_weights(self):
        self.q1.save_weights("q1.h5")
        self.q2.save_weights("q2.h5")

    def load_weights(self):
         # Initialise Experience Buffer D
        self.experience_buffer = ExperienceBuffer(self.max_buffer_len)
        self.q1.load_weights("q1.h5")
        self.q2.load_weights("q2.h5")