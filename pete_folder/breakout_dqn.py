# Try and do the gym's atari breakout

import gym
import numpy as np
import random
import math
import gc
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from dqn_utils import ReplayMemory, DataWithHistory


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

    def __init__(self, load_weights=None, exploratory_action_probability=0.05, adam_learning_rate=0.0001):
        """ Set up the FunctionApprox and policy so that training runs keep using the same.

        :param load_weights:
        :param exploratory_action_probability:
        """
        possible_actions = [0, 1, 2, 3]

        self.q_func = FunctionApprox(possible_actions, adam_learning_rate=adam_learning_rate)
        if load_weights is not None:
            self.q_func.load_weights(load_weights)
        self.policy = EGreedyPolicy(exploratory_action_probability, self.q_func, possible_actions)
        self.replay_memory = ReplayMemory(history=3)
        self.max_delta = None
        self.min_delta = None

    def train(self, env, num_episodes=10, step_size=1, discount_factor=0.9,
              save_weights=None ):
        # Implementation of the Sarsa algorithm
        # Algorithm parameters can be supplied to the train method, but defaults are:
        #   step_size (alpha) = 0.2
        #   discount_factor (gamma) = 0.9
        #   exploratory_action_probability (epsilon) > 0.15
        #   num_episodes = 150
        agent_rewards = []
        state_with_history = [DataWithHistory.empty_state() for i in range(4)]

        for episode in range(num_episodes):
            # Initialise S
            obs, info = env.reset()
            # TODO : add the history to the state.
            state = self.reformat_observation(obs)
            state_with_history.pop(0)
            state_with_history.append(state)
            # Choose A from S using policy derived from Q
            action, _ = self.policy.select_action(state_with_history)

            total_undiscounted_reward = 0
            terminated = False
            truncated = False
            self.max_delta = None
            self.min_delta = None

            steps = 0
            # bring the target and action value weights into sync after this many steps.
            clone_weights_count = 10

            while not terminated and not truncated:

                # Take action A, observe R, S'
                obs, reward, terminated, truncated, info = env.step(action)
                # TODO : add the history to the state.
                next_state = self.reformat_observation(obs)
                state_with_history.pop(0)
                state_with_history.append(next_state)
                # Choose A' from S' using policy derived from q_func
                next_action, _ = self.policy.select_action(state_with_history)

                self.replay_memory.add(state, action, reward, next_state, terminated)
                total_undiscounted_reward += reward
                if terminated:
                    print(f"terminated in episode {episode} after {steps+1} steps. "
                          f"Total reward {total_undiscounted_reward}")

                state, action = next_state, next_action

                # Replay steps from the memory to update the function approximation (q_func)
                self.replay_steps(discount_factor, step_size)

                clone_weights_count -= 1
                if clone_weights_count <= 0:
                    # every n steps clone the weights from the value to the target
                    self.q_func.clone_weights()
                    clone_weights_count = 10

                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

            agent_rewards.append(total_undiscounted_reward)

        if save_weights is not None:
            self.q_func.save_weights(save_weights)

        print(f"max delta = {self.max_delta}, min delta = {self.min_delta}")
        return agent_rewards

    def replay_steps(self, discount_factor, step_size, replay_num=3):
        """ Select items from the replay_memory and use them to update the q_func, value function approximation.

        :param discount_factor:
        :param step_size:
        :param replay_num: Number of random steps to replay - currently includes the latest step too.
        :return:
        """
        batch = self.replay_memory.get_batch(replay_num)
        # TODO : seems useful to add the latest too, but is it really
        batch.append(self.replay_memory.get_last_item())
        for data_item in batch:

            # Choose A' from S' using policy derived from q_func
            s = data_item.get_state()
            a = data_item.get_action()
            r = data_item.get_reward()
            next_s = data_item.get_next_state()
            # next_a, _ = self.policy.select_action(next_s)

            q_s_a = self.q_func.get_value(s, a)
            # discounted_next_q_s_a = discount_factor * self.q_func.get_target_value(next_s, next_a)
            discounted_next_q_s_a = discount_factor * self.q_func.get_max_target_value(next_s)
            if data_item.is_terminated():
                delta = step_size * r
            else:
                delta = step_size * (r + discounted_next_q_s_a - q_s_a)
            #
            if self.max_delta is None:
                self.max_delta = delta
            else:
                self.max_delta = max(self.max_delta, delta)
            if self.min_delta is None:
                self.min_delta = delta
            else:
                self.min_delta = min(self.min_delta, delta)
            self.q_func.update(a, s, delta)

    def reformat_observation(self, obs):
        # Change from array of 128 values of 0-255 to 16x8 of 0-1
        new_shape = np.array(obs).reshape(8, 16)
        return new_shape / 256

    def release_memory(self):
        """ The keras models created in FunctionApprox seem to hang onto memory even after they've gone out of scope.
        Try and force the clean up of the memory."""
        self.q_func.release_memory()


class FunctionApprox:

    def __init__(self, actions, update_batch_size=16, adam_learning_rate=0.0001):
        self.actions = actions
        self.q_hat = self.build_cnn(adam_learning_rate)
        self.q_hat_target = self.build_cnn(adam_learning_rate)
        self.clone_weights()
        self.update_batch_size = update_batch_size
        self.batch = []

    def release_memory(self):
        """ The keras models created seem to hang onto memory even after they've gone out of scope. Try and force
        the clean up of the memory."""
        del self.q_hat
        del self.q_hat_target
        gc.collect()
        K.clear_session()

    def save_weights(self, file_name):
        self.q_hat.save_weights(file_name)

    def load_weights(self, file_name):
        self.q_hat.load_weights(file_name)
        self.q_hat_target.load_weights(file_name)

    def clone_weights(self):
        # Copy the weights from action_value network to the target action_value network
        self.q_hat_target.set_weights(self.q_hat.get_weights())

    def build_cnn(self, adam_learning_rate):
        # Crete CNN model to predict actions for states.

        cnn = Sequential()
        # TODO : Find the best arrangement for the ConvNet
        cnn.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=(8, 16, 4)))
        cnn.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dense(10, activation='relu'))
        cnn.add(Dense(4, activation=None))
        cnn.summary()

        # compile the model
        optimizer = keras.optimizers.Adam(learning_rate=adam_learning_rate)
        cnn.compile(loss="mean_squared_error", optimizer=optimizer)

        return cnn

    def transpose_states(self, states):
        # states is a 4D array (N, X,Y,Z) with
        # N = number of states,
        # X = state and history, for CNN we need to transpose it to (N, Y,Z,X)
        # and also add another level.
        return np.transpose(np.array(states), (0, 2, 3, 1))

    def get_value(self, state, action):
        prediction = self.q_hat.predict_on_batch(self.transpose_states([state]))
        return prediction[0][action]

    def get_target_value(self, state, action):
        prediction = self.q_hat_target.predict_on_batch(self.transpose_states([state]))
        return prediction[0][action]

    def get_max_target_value(self, state):
        prediction = self.q_hat_target.predict_on_batch(self.transpose_states([state]))
        return max(prediction[0])

    def best_action_for(self, state):
        prediction = self.q_hat.predict_on_batch(self.transpose_states([state]))
        return np.argmax(prediction[0])

    def update(self, action, state, delta):
        # do the update in batches
        if len(self.batch) < self.update_batch_size:
            self.batch.append((action, state, delta))
            return

        # TODO : work out how to handle multiple states (i.e. the history) correctly
        states = np.array([s for (a, s, d) in self.batch])
        states = self.transpose_states(states)
        # get current prediction
        predictions = self.q_hat.predict_on_batch(states)

        # update values for specified actions
        for (action, _, delta), prediction in zip(self.batch, predictions):
            prediction[action] = prediction[action] + delta

        self.q_hat.train_on_batch(states, predictions)

        # clear the batch.
        self.batch = []

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


import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--render", type=float, default=None, help="Enter human to see the game played")
    parser.add_argument("--step_size", type=float, default=0.2, help="Step size when calculating the loss")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for Adam optimiser")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon for e-greedy policy")
    parser.add_argument("--agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per agent")
    parser.add_argument("--load_weights", default=None, help="Load weights from specified file")
    parser.add_argument("--save_weights", default=None, help="Save weights when agent finishes to specified file")


    args = parser.parse_args()

    # env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
    # env = gym.make("ALE/Breakout-v5", obs_type="ram")
    # env = gym.make("ALE/Breakout-v5", render_mode="human")
    best_reward = 0
    timer = Timer()
    timer.start("total time")
    if args.render == 'human':
        env = gym.make("ALE/Breakout-v5", obs_type="ram", render_mode="human")
    else:
        env = gym.make("ALE/Breakout-v5", obs_type="ram")

    agent = AgentBreakoutDqn(load_weights=args.load_weights,
                             exploratory_action_probability=args.epsilon,
                             adam_learning_rate=args.learning_rate)
    for i in range(args.agents):
        inner_timer = Timer()
        inner_timer.start("Agent run")
        print(f"\n{i+1}: run episodes")

        rewards = agent.train(env, num_episodes=args.episodes, save_weights=args.save_weights)
        # print(rewards)
        max_reward = max(rewards)
        total_rewards = sum(rewards)
        print(f"Total rewards from this run : {total_rewards}")
        best_reward = max(best_reward, max_reward)
        print(f"Best_reward overall = {best_reward}")

        inner_timer.end("Agent run")
        inner_timer.display()

    env.close()
    agent.release_memory()
    timer.end("total time")
    timer.display()


if __name__ == '__main__':
    main()