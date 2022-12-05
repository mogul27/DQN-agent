# Try and do the gym's atari breakout

import gym
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from collections import deque


class ReplayMemory:
    """ Maintain a memory of states and rewards from previous experience.

    The replay_memory simply maintains a list of all the (state, action, reward, next_state) combinations encountered
    while playing the game.
    It can then return a random selection from this list.
    As the list keeps being appended to, the random selections will eventually follow the same probability
    distribution as the searching for the goal sees.
    """
    def __init__(self):
        # TODO : Maybe cap the size of the file data?
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


class EGreedyPolicy:
    """ Assumes every state has the same possible actions.
    """

    def __init__(self, explore_probability, q_func, possible_actions):
        """ e-greedy policy based on the supplied q_table.

        :param explore_probability: small epsilon for the e-greedy policy. This is the probability that we'll
                                    randomly select an action, rather than picking the best.
        :param q_func: Approximates q values for state action pairs so we can select the best action.
        """
        self.q_func = q_func
        self.explore_probability = explore_probability
        self.possible_actions = possible_actions

    def select_action(self, state):
        """ The EGreedy policy selects the best action the majority of the time. However, a random action is chosen
        explore_probability amount of times.

        :param state: The sate to pick an action for
        :return: tuple (selected action, True if greedy selection | False if random selection)
        """
        # Select an action for the state, use the best action most of the time.
        # However, with probability of explore_probability, select an action at random.
        if np.random.uniform() < self.explore_probability:
            return random.choice(self.possible_actions), False

        return self.q_func.best_action_for(state), True


class AgentBreakoutDqn:

    def train(self, env, num_episodes=10, step_size=0.5, discount_factor=0.9, exploratory_action_probability=0.05,
              save_weights=False, load_weights=False):
        # Implementation of the Sarsa algorithm
        # Algorithm parameters can be supplied to the train method, but defaults are:
        #   step_size (alpha) = 0.2
        #   discount_factor (gamma) = 0.9
        #   exploratory_action_probability (epsilon) > 0.15
        #   num_episodes = 150

        possible_actions = [0, 1, 2, 3]
        agent_rewards = []

        q_func = FunctionApprox(possible_actions)
        if load_weights:
            q_func.load_weights()
        policy = EGreedyPolicy(exploratory_action_probability, q_func, possible_actions)
        replay_memory = ReplayMemory()

        display_at = 5
        best_reward = -1000000

        for episode in range(num_episodes):
            # Initialise S
            obs, info = env.reset()
            state = self.reformat_observation(obs)
            total_undiscounted_reward = 0
            terminated = False
            truncated = False
            # Choose A from S using policy derived from Q
            action, _ = policy.select_action(state)
            steps = 0
            clone_weights_count = 20

            while not terminated and not truncated:
                # Take action A, observe R, S'
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = self.reformat_observation(obs)
                total_undiscounted_reward += reward
                if terminated:
                    print(f"terminated in episode {episode} after {steps+1} steps. Total reward {total_undiscounted_reward}")

                # Choose A' from S' using policy derived from q_func
                next_action, _ = policy.select_action(next_state)

                q_s_a = q_func.get_value(state, action)
                discounted_next_q_s_a = discount_factor * q_func.get_target_value(next_state, next_action)
                delta = step_size * (reward + discounted_next_q_s_a - q_s_a)
                #
                q_func.update(action, state, delta)

                state, action = next_state, next_action

                clone_weights_count -= 1
                if clone_weights_count <= 0:
                    # every 30 steps clone the weights from the value to the target
                    q_func.clone_weights()
                    clone_weights_count = 30


                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

            agent_rewards.append(total_undiscounted_reward)
            best_reward = max(best_reward, total_undiscounted_reward)

            # display_at -= 1
            # if display_at <= 0:
            #     print(f"{episode+1}: best_reward = {best_reward}")
            #     display_at = 5

        # print(f"Finished: best_reward = {best_reward}")
        if save_weights:
            q_func.save_weights()

        return agent_rewards

    def reformat_observation(self, obs):
        # Change from array of 128 values of 0-255 to 16x8 of 0-1
        new_shape = np.array([
            obs[0:16],
            obs[16:32],
            obs[32:48],
            obs[48:64],
            obs[64:80],
            obs[80:96],
            obs[96:112],
            obs[112:128],
        ])
        return new_shape / 256

class FunctionApprox:

    def __init__(self, actions, num_tiles=4, tile_sections=10):
        self.actions = actions
        self.theta = {action: np.zeros(num_tiles*tile_sections*tile_sections) for action in actions}
        self.tiles = []
        self.q_hat = self.build_cnn()
        self.q_hat_target = self.build_cnn()
        self.clone_weights()

    def save_weights(self):
        self.q_hat.save_weights("q_hat.h5")
        self.q_hat_target.save_weights("q_hat_target.h5")

    def load_weights(self):
        self.q_hat.load_weights("q_hat.h5")
        self.q_hat_target.load_weights("q_hat_target.h5")

    def clone_weights(self):
        # Copy the weights from action_value network to the target action_value network
        self.q_hat_target.set_weights(self.q_hat.get_weights())

    def build_cnn(self):
        # Crete CNN model to predict actions for states.

        cnn = Sequential()
        # TODO : increase number of layers, maybe to 32
        cnn.add(Conv2D(16, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=(8, 16, 1)))
        cnn.add(Conv2D(16, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        cnn.add(Flatten())
        cnn.add(Dense(32, activation='relu'))
        cnn.add(Dense(4, activation='relu'))
        # cnn.summary()

        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return cnn

    def extract_features_for_state(self, state):
        (position, velocity) = state
        tile_features = [tile.extract_feature(position, velocity) for tile in self.tiles]
        return np.concatenate(tile_features)

    def get_value(self, state, action):
        prediction = self.q_hat.predict(np.array([state]), verbose=False)
        return prediction[0][action]

    def get_target_value(self, state, action):
        prediction = self.q_hat_target.predict(np.array([state]), verbose=False)
        return prediction[0][action]

    def best_action_for(self, state):
        prediction = self.q_hat.predict(np.array([state]), verbose=False)
        return np.argmax(prediction[0])

    def update(self, action, state, delta):
        # get current prediction
        prediction = self.q_hat.predict(np.array([state]), verbose=False)

        # update value for specified action
        prediction[0][action] = prediction[0][action] + delta

        self.q_hat.fit(np.array([state]), np.array(prediction), epochs=1, batch_size=1, verbose=False)

import time
class Timer:

    def __init__(self):
        self.event_timer = {}
        self.cummulative_times = {}

    def start(self, name):
        self.event_timer[name] = time.time()

    def end(self, name):
        if name in self.event_timer:
            if name not in self.cummulative_times:
                self.cummulative_times[name] = 0.0
            self.cummulative_times[name] += time.time() - self.event_timer[name]

    def display(self):
        for name, elapsed in self.cummulative_times.items():
            print(f"{name} : {elapsed:0.1f} seconds")


def main():

    # env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
    # env = gym.make("ALE/Breakout-v5", obs_type="ram")
    # env = gym.make("ALE/Breakout-v5", render_mode="human")
    best_reward = 0
    timer = Timer()
    timer.start("total time")
    # env = gym.make("ALE/Breakout-v5", obs_type="ram")
    env = gym.make("ALE/Breakout-v5", obs_type="ram", render_mode="human")
    for i in range(1):
        inner_timer = Timer()
        inner_timer.start("Agent run")
        print(f"\n{i+1}: run episodes")

        agent = AgentBreakoutDqn()
        rewards = agent.train(env, num_episodes=5, load_weights=False, save_weights=False)
        # print(rewards)
        max_reward = max(rewards)
        total_rewards = sum(rewards)
        print(f"Total rewards from this run : {total_rewards}")
        best_reward = max(best_reward, max_reward)
        print(f"Best_reward overall = {best_reward}")

        inner_timer.end("Agent run")
        inner_timer.display()

    env.close()
    timer.end("total time")
    timer.display()


if __name__ == '__main__':
    main()