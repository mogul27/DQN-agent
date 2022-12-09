import gym
import numpy as np
import random
from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.losses import Huber
from collections import deque


class ExperienceBuffer:
    """ Maintain a memory of states and rewards from previous experience.

    Stores (state, action, reward, next_state) combinations encountered while playing the game up
    to a limit N
    Return random selection from the list
    """

    def __init__(self, max_buffer_len: int):
        self.data = deque()
        self.size = 0
        self.max_len = max_buffer_len

    def add(self, state, action, reward, next_state, terminal):
        if self.size >= self.max_len:
            # don't grow bigger, just lose one off the front.
            self.data.popleft()
        else:
            self.size += 1
        self.data.append((state, action, reward, next_state, terminal))

    def get_random_data(self, batch_size=1):
        # get a random batch of data
        #
        # :param batch_size: Number of data elements - default 1
        # :return: list of tuples of (state, action, reward, next_state)

        return random.choices(self.data, k=batch_size)


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
        action_value_network.add(Dense(num_actions))

        # compile the model
        optimiser = Adam(learning_rate=0.01)
        action_value_network.compile(optimizer=optimiser, loss=Huber(), metrics=['accuracy'])

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

    def get_q2_preds(self, network_input):
        """Retrieve the best action to take in given state"""
        q_vals = self.q2.predict_on_batch(network_input)
        best_action = np.argmax(q_vals[0]) 
        best_action_val = q_vals[0][best_action]

        return q_vals, best_action_val

    def get_q1_action_value(self, network_input):
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
    
    def save_weights(self):
        self.q1.save_weights("q1.h5")
        self.q2.save_weights("q2.h5")

    def load_weights(self):
         # Initialise Experience Buffer D
        self.experience_buffer = ExperienceBuffer(self.max_buffer_len)
        self.q1.load_weights("q1.h5")
        self.q2.load_weights("q2.h5")

def main(load_weights=False):

    # Set algorithm parameters
    minibatch_size = 32 #32
    experience_buffer_size = 20000 #20000
    max_episodes = 500 # 100
    epsilon = 1
    epsilon_decay = 0.9/20000 # decay epsilon to 0.1 over first 100000 frames
    final_epsilon = 0.1
    gamma = 0.99
    c = 1000 # How many steps until update parameters of networks to match

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
    
    # Initialise a new DQNAgent
    agent = DQNAgent(epsilon=epsilon, max_buffer_len=experience_buffer_size, gamma=gamma)

    if load_weights:
        agent.load_weights()
    else:
        agent.initialise_network_buffer(num_actions, state_dims)

    # Set terminal to False initially for looping
    terminal=False
    # Fill replay buffer sith initial random wandering
    for _ in range(experience_buffer_size):
        action = np.random.choice(possible_actions)
        next_state, reward, terminal, _, _ = wrapped_env.step(action)
        # Add experience to the buffer
        agent.experience_buffer.add(prev_state, action, reward, next_state, terminal)
        prev_state = next_state

        if terminal:
            prev_state = wrapped_env.reset()

    # Initialise episode monitoring (rewards, num_episodes)
    episode_rewards = []
    total_episode_rewards= []
    episode_counter = 0
    step_number = []

    for episode in range(max_episodes):

        # Reset environment now that replay buffer filled or new episode started
        env = gym.make("ALE/DemonAttack-v5", render_mode='human', frameskip=1)
        wrapped_env = gym.wrappers.AtariPreprocessing(env)
        prev_state = wrapped_env.reset()
        terminal = False

        step=0
        rewards=[]

        while not terminal:
        
            # Account for different shape of initial state
            if step == 0:
                # Reshape prev_state array data to be passed into network
                network_input = prev_state[0].reshape(1, 84, 84, 1)
            else:
                network_input = prev_state.reshape(1, 84, 84, 1)
            action = agent.epsilon_greedy_selection(num_actions,
                                                    possible_actions, network_input)
            next_state, reward, terminal, _, _ = wrapped_env.step(action)
            rewards.append(reward)

            agent.experience_buffer.add(prev_state, action, reward, next_state, terminal)
            prev_state = next_state

            minibatch = agent.experience_buffer.get_random_data(minibatch_size)

            network_input_batch = []
            target_preds_batch = []

            for experience in minibatch:
                
                # Unpack experience
                # Label with exp to avoid overwriting current state
                prev_state_exp, action_exp, reward_exp, next_state_exp, terminal_exp = experience

                # Reshape next_state to be passed into network
                network_input = next_state_exp.reshape(1, 84, 84, 1)
                
                # Retrieve q2 predictions and value for the best action
                target_preds, best_action_val = agent.get_q2_preds(network_input)
                
                if terminal_exp:
                    target_preds[0][action_exp] = reward_exp
                else:
                    # Retrieve best possible action according to target network
                    target_preds[0][action_exp] = reward_exp + gamma*best_action_val


                # Reshape prev_state to be passed into network
                # Previous states sampled may be a different shape/type

                if type(prev_state_exp) == tuple:
                    network_input = prev_state_exp[0].reshape(1, 84, 84, 1)
                else:
                    network_input = prev_state_exp.reshape(1, 84, 84, 1)

                # use [0] to avoid batch_size being added as an extra_sim
                network_input_batch.append(network_input[0])
                target_preds_batch.append(target_preds)

            network_input_batch = np.array(network_input_batch)
            target_preds_batch=np.array(target_preds_batch)
            agent.q1.train_on_batch(network_input_batch, target_preds_batch)
            
            # Every c timesteps
            if step % c == 0:
                agent.q2.set_weights(agent.q1.get_weights())
                print(np.mean(rewards))
                print("Step:", step)
            
            step += 1
            
            if epsilon > final_epsilon:
                epsilon = epsilon - epsilon_decay
            
            

        episode_rewards.append(np.mean(rewards))
        total_episode_rewards.append(sum(rewards))
        episode_counter += 1
        step_number.append(step)
        print("episode {} complete. Total episode Reward: {}".format(episode_counter, sum(rewards)))

        # Save weights and write episode reward
        agent.save_weights()

        if episode_counter == 1:
            with open('rewards.txt', 'w') as reward_txt:
                reward_txt.write("Episode: {}, Total Reward: {}, Steps: {}".format(
                                episode_counter, sum(rewards), step))
        else:
            with open("rewards.txt", "a") as reward_txt:
                # Append next epsiode reward at the end of file
                reward_txt.write("\nEpisode: {}, Total Reward: {}, Steps: {}".format(
                                episode_counter, sum(rewards), step))

    print("Episode average rewards: ", episode_rewards)
    print("Max episode average Reward:", max(episode_rewards))
    print("Episode total rewards: ", total_episode_rewards)

        

# y is target because that's what the target network does (actual answer)
# y_hat is your attempt at it

            # OFf-policy so replacing says here's what you could do in that state
            # It's S' so it's the best possible path from that action
            # update to move closer to the value possible from the next state if you take that action

main()