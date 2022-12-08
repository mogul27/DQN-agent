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
            action = random.choice(self.possible_actions)
        else:
            action = self.q_func.best_action_for(state)

        # decay epsilon for next time
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        return action


class AgentBreakoutDqn:

    def __init__(self, load_weights=None, epsilon=None, epsilon_decay=None, epsilon_min=None,
                 adam_learning_rate=None, step_size=None, discount_factor=None):
        """ Set up the FunctionApprox and policy so that training runs keep using the same.

        :param load_weights:
        :param exploratory_action_probability:
        """
        # Set default values
        if epsilon is None:
            epsilon = 1.0
        if epsilon_decay is None:
            epsilon_decay = 0.000001
        if epsilon_min is None:
            epsilon_min = 0.05
        if adam_learning_rate is None:
            adam_learning_rate = 0.00025
        if step_size is None:
            step_size = 1.0
        if discount_factor is None:
            discount_factor = 0.99

        self.step_size = step_size
        self.discount_factor = discount_factor
        self.adam_learning_rate = adam_learning_rate

        possible_actions = [0, 1, 2, 3]

        self.q_func = FunctionApprox(possible_actions, adam_learning_rate=adam_learning_rate)
        if load_weights is not None:
            self.q_func.load_weights(load_weights)
        self.policy = EGreedyPolicy(epsilon, self.q_func, possible_actions, epsilon_decay, epsilon_min)
        self.play_policy = EGreedyPolicy(0.05, self.q_func, possible_actions)
        self.replay_memory = ReplayMemory(history=3)
        self.max_delta = None
        self.min_delta = None

    def play(self, env, start_action=1, num_lives=5):
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
        action = start_action
        last_action = -1
        repeated_action_count = 0
        life_lost = True

        while not terminated and not truncated:
            if action == last_action:
                repeated_action_count += 1
                # check it doesn't get stuck
                if repeated_action_count > 250:
                    print(f"Play with greedy policy has probably got stuck - action {action} repeated 250 times")
                    break
            else:
                repeated_action_count = 0

            last_obs = obs
            life_lost = False
            obs, reward, terminated, truncated, info = env.step(action)
            if 'lives' in info:
                if info['lives'] < num_lives:
                    num_lives = info['lives']
                    life_lost = True

            total_reward += reward

            next_state = self.reformat_observation(obs, last_obs)
            state_with_history.pop(0)
            state_with_history.append(next_state)
            last_action = action
            if life_lost:
                action = start_action
            else:
                action = self.play_policy.select_action(state_with_history)

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        return total_reward

    def train(self, env, num_episodes=10, save_weights=None, start_action=1, num_lives=5):
        agent_rewards = []
        frame_count = 0
        state_with_history = [DataWithHistory.empty_state() for i in range(4)]

        for episode in range(num_episodes):
            # Initialise S
            obs, info = env.reset()
            # TODO : add the history to the state.
            state = self.reformat_observation(obs, obs)
            state_with_history.pop(0)
            state_with_history.append(state)
            # Choose A from S using policy derived from Q
            if start_action is None:
                action = self.policy.select_action(state_with_history)
            else:
                action = start_action

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
                last_obs = obs
                life_lost = False
                obs, reward, terminated, truncated, info = env.step(action)
                if 'lives' in info:
                    if info['lives'] < num_lives:
                        num_lives = info['lives']
                        life_lost = True

                # TODO : add the history to the state.
                next_state = self.reformat_observation(obs, last_obs)
                state_with_history.pop(0)
                state_with_history.append(next_state)

                if life_lost:
                    next_action = start_action
                else:
                    # Choose A' from S' using policy derived from q_func
                    next_action = self.policy.select_action(state_with_history)

                self.replay_memory.add(state, action, reward, next_state, terminated or life_lost)
                total_undiscounted_reward += reward
                if terminated:
                    print(f"finished episode {episode+1} after {steps+1} steps. "
                          f"Total reward {total_undiscounted_reward}")

                state, action = next_state, next_action

                # Replay steps from the memory to update the function approximation (q_func)
                self.replay_steps()

                clone_weights_count -= 1
                if clone_weights_count <= 0:
                    # every n steps clone the weights from the value to the target
                    self.q_func.clone_weights()
                    clone_weights_count = 10

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

    def replay_steps(self, replay_num=3):
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
            # next_a = self.policy.select_action(next_s)

            q_s_a = self.q_func.get_value(s, a)
            # discounted_next_q_s_a = discount_factor * self.q_func.get_target_value(next_s, next_a)
            discounted_next_q_s_a = self.discount_factor * self.q_func.get_max_target_value(next_s)
            if data_item.is_terminated():
                delta = self.step_size * r
            else:
                delta = self.step_size * (r + discounted_next_q_s_a - q_s_a)
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


def run(render=None, training_cycles=1, num_episodes=1, epsilon=None, epsilon_decay=None, epsilon_min=None,
        learning_rate=None, step_size=None, discount_factor=None,
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
                             step_size=step_size, discount_factor=discount_factor)
    for i in range(training_cycles):
        timer.start("Training cycle")

        print(f"\nTraining cycle {i+1} of {training_cycles}:")
        print(f"Start epsilon = {agent.policy.epsilon:0.5f}"
              f", step_size = {agent.step_size}, discount_factor = {agent.discount_factor}"
              f", learning_rate = {agent.adam_learning_rate}")

        rewards, frame_count = agent.train(env, num_episodes=num_episodes, save_weights=save_weights)

        timer.stop("Training cycle")
        training_time = int(timer.event_times["Training cycle"])
        cumulative_training_time = int(timer.cumulative_times["Training cycle"])

        # print some useful info
        total_frames += frame_count
        print(f"Frames this cycle = {frame_count}, in {training_time} seconds")
        print(f"Total frames over all cycles = {total_frames}, in {cumulative_training_time} seconds")
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
    run( render='human', training_cycles=2, num_episodes=1)


if __name__ == '__main__':
    main()