# Try and do the gym's atari breakout

import gym
import numpy as np
import random
import math
import gc
import cv2
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.losses import Huber
from dqn_utils import ReplayMemory, DataWithHistory, Timer


class EGreedyPolicy:
    """ Assumes every state has the same possible actions.
    """

    def __init__(self, epsilon, q_func, possible_actions, epsilon_decay=None, epsilon_min=None):
        """ e-greedy policy based on the supplied q_table.

        :param epsilon: small epsilon for the e-greedy policy. This is the probability that we'll
                                    randomly select an action, rather than picking the best.
        :param q_func: Approximates q values for state action pairs so we can select the best action.
        :param possible_actions: actions to be selected from with epsilon probability
        :param epsilon_decay: the amount to reduce epsilon by after select action has been performed
        :param epsilon_min: the min value epsilon can be after decay
        """
        self.q_func = q_func
        self.epsilon = epsilon
        self.possible_actions = possible_actions
        if epsilon_decay is None:
            self.epsilon_decay = 0
            self.epsilon_min = epsilon
        else:
            self.epsilon_decay = epsilon_decay
            if epsilon_min is None:
                self.epsilon_min = 0
            else:
                self.epsilon_min = epsilon_min

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
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        return action

    def random_action(self):
        return random.choice(self.possible_actions)


class AgentBreakoutDqn:

    def __init__(self, load_weights=None, epsilon=None, epsilon_decay=None, epsilon_min=None,
                 adam_learning_rate=None, discount_factor=None,
                 start_action=1, start_lives=5):
        """ Set up the FunctionApprox and policy so that training runs keep using the same.

        :param load_weights:
        :param exploratory_action_probability:
        """
        # Set default values
        if epsilon is None:
            epsilon = 1.0
        if epsilon_decay is None:
            epsilon_decay = 0.0000025
        if epsilon_min is None:
            epsilon_min = 0.1
        if adam_learning_rate is None:
            adam_learning_rate = 0.00025
        if discount_factor is None:
            discount_factor = 0.99

        self.discount_factor = discount_factor
        self.adam_learning_rate = adam_learning_rate
        self.start_action = start_action
        self.start_lives = start_lives
        self.num_lives = start_lives

        possible_actions = [0, 1, 2, 3]

        self.q_func = FunctionApprox(possible_actions, adam_learning_rate=adam_learning_rate)
        if load_weights is not None:
            self.q_func.load_weights(load_weights)
        self.policy = EGreedyPolicy(epsilon, self.q_func, possible_actions, epsilon_decay, epsilon_min)
        self.play_policy = EGreedyPolicy(0.05, self.q_func, possible_actions)
        self.replay_memory = ReplayMemory(max_len=100000, history=3)
        self.max_delta = None
        self.min_delta = None

    def take_step(self, env, action, skip=3):
        life_lost = False
        obs, reward, terminated, truncated, info = env.step(action)
        if 'lives' in info:
            if info['lives'] < self.num_lives:
                self.num_lives = info['lives']
                life_lost = True
        skippy = skip
        while reward == 0 and not terminated and not truncated and not life_lost and skippy > 0:
            skippy -= 1
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

        return obs, reward, terminated, truncated, info, life_lost

    def init_replay_memory(self, env, initial_size=50000):

        while self.replay_memory.size < initial_size:

            obs, info = env.reset()

            state = self.reformat_observation(obs, obs)
            action = self.start_action

            self.num_lives = self.start_lives

            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:

                last_obs = obs
                obs, reward, terminated, truncated, info, life_lost = self.take_step(env, action)

                next_state = self.reformat_observation(obs, last_obs)
                self.replay_memory.add(state, action, reward, next_state, terminated or life_lost)
                if self.replay_memory.size >= initial_size:
                    # Replay memory is the required size, so break out.
                    break

                if life_lost:
                    action = self.start_action
                else:
                    action = self.policy.random_action()

                state = next_state

                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

    def play(self, env):
        """ play a single episode using a greedy policy """
        total_reward = 0
        state_with_history = [DataWithHistory.empty_state() for i in range(4)]

        # Init game
        obs, info = env.reset()
        state = self.reformat_observation(obs, obs)
        state_with_history.pop(0)
        state_with_history.append(state)

        terminated = False
        truncated = False
        steps = 0
        action = self.start_action
        last_action = -1
        repeated_action_count = 0
        self.num_lives = self.start_lives
        action_frequency = {
            0: 0, 1: 0, 2: 0, 3: 0
        }

        while not terminated and not truncated:
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

            next_state = self.reformat_observation(obs, last_obs)
            state_with_history.pop(0)
            state_with_history.append(next_state)
            last_action = action
            if life_lost:
                action = self.start_action
            else:
                action = self.play_policy.select_action(state_with_history)

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        print(f"Action frequency: {action_frequency}")
        return total_reward

    def train(self, env, num_episodes=10, save_weights=None):
        agent_rewards = []
        frame_count = 0
        state_with_history = [DataWithHistory.empty_state() for i in range(4)]

        for episode in range(num_episodes):
            # sync value and target weights at the start of an episode
            self.q_func.clone_weights()

            # Initialise S
            obs, info = env.reset()

            state = self.reformat_observation(obs, obs)
            state_with_history.pop(0)
            state_with_history.append(state)

            action = self.start_action

            self.num_lives = self.start_lives
            total_undiscounted_reward = 0
            terminated = False
            truncated = False

            self.max_delta = None
            self.min_delta = None

            steps = 0

            while not terminated and not truncated:

                # Take action A, observe R, S'
                last_obs = obs
                obs, reward, terminated, truncated, info, life_lost = self.take_step(env, action)

                next_state = self.reformat_observation(obs, last_obs)
                state_with_history.pop(0)
                state_with_history.append(next_state)

                if life_lost:
                    next_action = self.start_action
                else:
                    # Choose A' from S' using policy derived from q_func
                    next_action = self.policy.select_action(state_with_history)

                self.replay_memory.add(state, action, reward, next_state, terminated or life_lost)
                total_undiscounted_reward += reward
                if terminated:
                    print(f"finished episode {episode+1} after {steps+1} steps. "
                          f"Total reward {total_undiscounted_reward}")

                state, action = next_state, next_action

                # TODO : make the replay steps less frequent?
                # Replay steps from the memory to update the function approximation (q_func)
                self.replay_steps()

                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

            frame_count += steps
            agent_rewards.append(total_undiscounted_reward)

        if save_weights is not None:
            self.q_func.save_weights(save_weights)

        print(f"max delta = {self.max_delta}, min delta = {self.min_delta}")
        return agent_rewards, frame_count

    def replay_steps(self, replay_num=32):
        """ Select items from the replay_memory and use them to update the q_func, value function approximation.

        :param replay_num: Number of random steps to replay - currently includes the latest step too.
        """
        batch = self.replay_memory.get_batch(replay_num)

        # process the batch - get the values and target values in single calls
        states = [data_item.get_state() for data_item in batch]
        next_states = [data_item.get_next_state() for data_item in batch]
        qsa_action_values = self.q_func.get_all_action_values(states)
        next_state_action_values = self.q_func.get_max_target_values(next_states)
        discounted_next_qsa_values = self.discount_factor * next_state_action_values

        for data_item, qsa_action_value, discounted_next_qsa_value in zip(batch, qsa_action_values, discounted_next_qsa_values):
            a = data_item.get_action()
            s = data_item.get_state()
            r = data_item.get_reward()
            if data_item.is_terminated():
                y = r
            else:
                y = r + discounted_next_qsa_value

            delta = qsa_action_value[a] - y

            # update the action value to move closer to the target
            qsa_action_value[a] = y
            self.q_func.update(s, qsa_action_value)

            if self.max_delta is None:
                self.max_delta = delta
            else:
                self.max_delta = max(self.max_delta, delta)
            if self.min_delta is None:
                self.min_delta = delta
            else:
                self.min_delta = min(self.min_delta, delta)

    def reformat_observation(self, obs, last_obs):
        # take the max from obs and last_obs to reduce odd/even flicker that Atari 2600 has
        merged_obs = np.maximum(obs, last_obs)
        # reduce merged greyscalegreyscale from 210,160 down to 84,84
        return cv2.resize(merged_obs, (84, 84), interpolation=cv2.INTER_AREA)

    def release_memory(self):
        """ The keras models created in FunctionApprox seem to hang onto memory even after they've gone out of scope.
        Try and force the clean up of the memory."""
        self.q_func.release_memory()


class FunctionApprox:

    def __init__(self, actions, update_batch_size=32, adam_learning_rate=0.0001):
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
        cnn.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
        cnn.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dense(4, activation=None))
        cnn.summary()

        # compile the model
        optimizer = keras.optimizers.Adam(learning_rate=adam_learning_rate)
        cnn.compile(loss=Huber(delta=1.0), optimizer=optimizer)

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


def run(render=None, training_cycles=1, num_episodes=1, epsilon=None, epsilon_decay=None, epsilon_min=None,
        learning_rate=None, discount_factor=None, replay_init_size=50000,
        load_weights=None, save_weights=None):

    # env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
    # env = gym.make("ALE/Breakout-v5", obs_type="ram")
    # env = gym.make("ALE/Breakout-v5", render_mode="human")
    best_reward = 0
    total_frames = 0
    timer = Timer()
    best_play_reward = 0

    if render == 'human':
        env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode="human", frameskip=1)
    else:
        env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=1)

    agent = AgentBreakoutDqn(load_weights=load_weights,
                             epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                             adam_learning_rate=learning_rate,
                             discount_factor=discount_factor)

    timer.start("Init replay memory")
    agent.init_replay_memory(env, replay_init_size)
    timer.stop("Init replay memory")
    replay_init_time = int(timer.event_times["Init replay memory"])
    print(f"Replay memory initialised with {agent.replay_memory.size} items in {replay_init_time} seconds")

    for i in range(training_cycles):
        timer.start("Training cycle")

        print(f"\nTraining cycle {i+1} of {training_cycles}:")
        print(f"Start epsilon = {agent.policy.epsilon:0.5f}"
              f", discount_factor = {agent.discount_factor}"
              f", learning_rate = {agent.adam_learning_rate}")

        rewards, frame_count = agent.train(env, num_episodes=num_episodes, save_weights=save_weights)

        timer.stop("Training cycle")
        training_time = int(timer.event_times["Training cycle"])
        cumulative_training_time = int(timer.cumulative_times["Training cycle"])

        # print some useful info
        total_frames += frame_count
        print(f"Frames this cycle = {frame_count}, in {training_time} seconds")
        print(f"Total frames over all cycles = {total_frames}, in {cumulative_training_time} seconds")
        if len(rewards) == 0:
            max_reward = 0
        else:
            max_reward = max(rewards)
        total_rewards = sum(rewards)
        print(f"Total training rewards from this cycle : {total_rewards}")
        best_reward = max(best_reward, max_reward)
        print(f"Best training reward overall = {best_reward}")


        print(f"Training cycle finished, time taken: {training_time:0.1f} secs")
        # play the game with greedy policy
        play_reward = agent.play(env)
        best_play_reward = max(best_play_reward, play_reward)
        print(f"Play reward = {play_reward}, best play reward overall = {best_play_reward}")

    env.close()
    agent.release_memory()
    del agent

    total_time = timer.cumulative_times["Training cycle"]
    total_mins = int(total_time // 60)
    total_secs = total_time - (total_mins * 60)
    print(f"\nRun finished. Total time taken: {total_mins} mins {total_secs:0.1f} secs")


def main():
    # run(render='human')
    run(
        render='human',
        # epsilon=0.5,
        # epsilon_decay=0.05,
        training_cycles=5,
        num_episodes=0,
        load_weights="breakout.h5",
        replay_init_size=0)
    # run(render='human', training_cycles=5, num_episodes=0, load_weights="batched_updates_2.h5",
    #     replay_init_size=0)
    # run(training_cycles=2, num_episodes=1)


if __name__ == '__main__':
    main()