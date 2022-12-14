# Try and do the gym's atari breakout

import gym
import numpy as np
import random
import math
import os
from pathlib import Path
import gc
import cv2
import keras
from keras import backend as K
from keras.models import Sequential
from keras import Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.losses import Huber
from keras.optimizers import Adam
from dqn_utils import ReplayMemory, DataWithHistory, Timer


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


class AgentClassicControlDqn:

    def __init__(self, load_weights=None, work_dir=None, epsilon=None, epsilon_decay_span=None, epsilon_min=None,
                 adam_learning_rate=None, discount_factor=None):
        """ Set up the FunctionApprox and policy so that training runs keep using the same.

        :param load_weights:
        :param exploratory_action_probability:
        """
        self.work_dir = work_dir
        if self.work_dir is not None:
            # location for files.
            self.work_dir = Path(work_dir)
            if not self.work_dir.exists():
                self.work_dir.mkdir(parents=True, exist_ok=True)

        # Set default values
        if epsilon is None:
            epsilon = 1.0
        if epsilon_decay_span is None:
            epsilon_decay_span = 50000
        if epsilon_min is None:
            epsilon_min = 0.1
        if adam_learning_rate is None:
            adam_learning_rate = 0.00025
        if discount_factor is None:
            discount_factor = 0.99

        self.discount_factor = discount_factor
        self.adam_learning_rate = adam_learning_rate
        self.num_lives = 0

        possible_actions = [0, 1, 2]

        self.q_func = FunctionApprox(possible_actions, adam_learning_rate=adam_learning_rate)
        if load_weights is not None:
            if self.work_dir is not None:
                load_weights = os.fspath(self.work_dir / load_weights)
            self.q_func.load_weights(load_weights)
        self.policy = EGreedyPolicy(epsilon, self.q_func, possible_actions, epsilon_decay_span, epsilon_min)
        self.play_policy = EGreedyPolicy(0.05, self.q_func, possible_actions)
        if self.work_dir is None:
            replay_memory_file = None
        else:
            replay_memory_file = os.fspath(self.work_dir / 'replay_memory.pickle')
        self.replay_memory = ReplayMemory(max_len=20000, history=0, file_name=replay_memory_file)
        self.max_delta = None
        self.min_delta = None
        self.total_episodes = 0

    def init_replay_memory(self, env, initial_size=5000):

        while self.replay_memory.size < initial_size:

            state, info = env.reset()

            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:

                action = self.policy.random_action()
                next_state, reward, terminated, truncated, info = env.step(action)

                self.replay_memory.add(state, action, reward, next_state, terminated)
                if self.replay_memory.size >= initial_size:
                    # Replay memory is the required size, so break out.
                    break

                state = next_state

                steps += 1
                if steps >= initial_size:
                    print(f"Break out as we've taken {steps} steps during replay memory initialisation. "
                          f"Something has probably gone wrong...")
                    break

        self.replay_memory.save()

    def play(self, env):
        """ play a single episode using a greedy policy """
        total_reward = 0

        # Init game
        state, info = env.reset()

        terminated = False
        truncated = False
        steps = 0
        last_action = -1
        repeated_action_count = 0
        action_frequency = {a: 0 for a in self.q_func.actions}

        while not terminated and not truncated:
            action = self.play_policy.select_action(state)
            if action == last_action:
                repeated_action_count += 1
                # check it doesn't get stuck
                if repeated_action_count > 1000:
                    print(f"Play with greedy policy has probably got stuck - action {action} repeated 1000 times")
                    break
            else:
                repeated_action_count = 0
            action_frequency[action] += 1

            state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            last_action = action

            steps += 1
            if steps >= 10000:
                print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                break

        print(f"Action frequency: {action_frequency}")
        return total_reward

    def train(self, env, num_episodes=10, save_weights=None):
        sync_weights_count = 0
        agent_rewards = []
        frame_count = 0

        for episode in range(num_episodes):


            # Initialise S
            state, info = env.reset()

            action = self.policy.select_action(state)

            total_undiscounted_reward = 0
            terminated = False
            truncated = False

            self.max_delta = None
            self.min_delta = None

            steps = 0

            while not terminated and not truncated:
                # sync value and target weights at the start of an episode
                sync_weights_count -= 1
                if sync_weights_count <= 0:
                    self.q_func.clone_weights()
                    sync_weights_count = 1000

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, info = env.step(action)

                # Choose A' from S' using policy derived from q_func
                next_action = self.policy.select_action(state)

                self.replay_memory.add(state, action, reward, next_state, terminated)
                total_undiscounted_reward += reward


                state, action = next_state, next_action

                # TODO : make the replay steps less frequent?
                # Replay steps from the memory to update the function approximation (q_func)
                self.replay_steps()

                steps += 1
                if steps >= 100000:
                    print(f"Break out as we've taken {steps} steps. Something has probably gone wrong...")
                    break

            self.total_episodes += 1
            print(f"finished episode {self.total_episodes} after {steps} steps. Total reward {total_undiscounted_reward}")
            frame_count += steps
            agent_rewards.append(total_undiscounted_reward)

        if save_weights is not None:
            if self.work_dir is not None:
                self.q_func.save_weights(self.work_dir / save_weights)
            else:
                self.q_func.save_weights(save_weights)

        self.replay_memory.save()

        print(f"max delta = {self.max_delta}, min delta = {self.min_delta}")
        return agent_rewards, frame_count

    def process_batch(self, replay_batch):
        """ Process the batch and update the q_func, value function approximation.

        :param state_action_batch: List of tuples with state, action, reward, next_state, terminated
        """
        # process the batch - get the values and target values in single calls
        # TODO : make this neater - vector based?
        state_action_batch = [data_item.data[0] for data_item in replay_batch]
        states = []
        next_states = []
        actions = []
        rewards = []
        for data_item in state_action_batch:
            states.append(data_item[0])
            next_states.append(data_item[3])
            actions.append(data_item[1])
            rewards.append(data_item[2])
        states = np.array(states)
        next_states = np.array(next_states)

        qsa_action_values = self.q_func.get_all_action_values(states)
        next_state_action_values = self.q_func.get_max_target_values(next_states)
        discounted_next_qsa_values = self.discount_factor * next_state_action_values
        updates = []
        min_d = None
        max_d = None
        min_q = None
        max_q = None
        # weights = self.q_func.get_value_network_weights()
        # print(f"worker {self.pid} : weights[0][0][0][0][0:5] {weights[0][0][0][0][0:5]}")
        # print(f"worker {self.pid} : weights[1][0] {weights[1][0]}")
        #
        # print(f"worker {self.pid} : a={actions} : r={rewards} : qsa={qsa_action_values} : qsa_next={discounted_next_qsa_values}")
        for data_item, qsa_action_value, discounted_next_qsa_value in zip(state_action_batch, qsa_action_values, discounted_next_qsa_values):
            s, a, r, next_s, terminated = data_item
            if terminated:
                y = r
            else:
                y = r + discounted_next_qsa_value

            delta = qsa_action_value[a] - y
            if min_d is None:
                min_d = delta
                max_d = delta
            else:
                min_d = min(delta, min_d)
                max_d = max(delta, max_d)

            if min_q is None:
                min_q = qsa_action_value[a]
                max_q = qsa_action_value[a]
            else:
                min_q = min(qsa_action_value[a], min_q)
                max_q = max(qsa_action_value[a], max_q)

            # update the action value to move closer to the target
            qsa_action_value[a] = y

            updates.append((s, qsa_action_value))

        losses = self.q_func.update_batch(updates)
        return losses, min_d, max_d, min_q, max_q

    def replay_steps(self, replay_num=32):
        """ Select items from the replay_memory and use them to update the q_func, value function approximation.

        :param replay_num: Number of random steps to replay - currently includes the latest step too.
        """
        batch = self.replay_memory.get_batch(replay_num)
        loss, min_d, max_d, min_q, max_q = self.process_batch(batch)
        # if min_delta is None:
        #     min_delta = min_d
        #     max_delta = max_d
        # else:
        #     min_delta = min(min_d, min_delta)
        #     max_delta = min(max_d, max_delta)
        # if min_q_value is None:
        #     min_q_value = min_q
        #     max_q_value = max_q
        # else:
        #     min_q_value = min(min_q, min_q_value)
        #     max_q_value = min(max_q, max_q_value)


class FunctionApprox:

    def __init__(self, actions, update_batch_size=32, adam_learning_rate=0.0001):
        self.actions = actions
        self.q_hat = self.build_neural_network(adam_learning_rate)
        self.q_hat_target = self.build_neural_network(adam_learning_rate)
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
        print(f"clone weights so target = value")
        self.q_hat_target.set_weights(self.q_hat.get_weights())

    def build_neural_network(self, adam_learning_rate):
        # Crete CNN model to predict actions for states.

        try:
            # TODO : give all the layers and models names to indicate worker / controller ?

            inputs = Input((2,))
            dense_1 = Dense(512, activation='relu')(inputs)
            dense_2 = Dense(64, activation='relu')(dense_1)
            outputs = Dense(len(self.actions), activation='linear')(dense_1)
            network = Model(inputs=inputs, outputs=outputs)

            network.summary()

            # compile the model
            network.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=adam_learning_rate))

            return network
        except Exception as e:
            print(f"failed to create model : {e}")

    def get_value(self, state, action):
        prediction = self.q_hat.predict_on_batch(np.expand_dims(state, axis=0))
        return prediction[0][action]

    def get_all_action_values(self, states):
        return self.q_hat.predict_on_batch(states)

    def get_target_value(self, state, action):
        prediction = self.q_hat_target.predict_on_batch(np.expand_dims(state, axis=0))
        return prediction[0][action]

    def get_max_target_value(self, state):
        prediction = self.q_hat_target.predict_on_batch(np.expand_dims(state, axis=0))
        return max(prediction[0])

    def get_max_target_values(self, states):
        predictions = self.q_hat_target.predict_on_batch(states)
        return predictions.max(axis=1)

    def best_action_for(self, state):
        prediction = self.q_hat.predict_on_batch(np.expand_dims(state, axis=0))
        return np.argmax(prediction[0])

    def update_batch(self, batch):
        # do the update in batches
        states = np.array([s for (s, new_action_value) in batch])
        new_action_values = np.array([new_action_value for (s, new_action_value) in batch])

        return self.q_hat.train_on_batch(states, new_action_values)

def register_gym_mods():
    gym.envs.registration.register(
        id='MountainCarMyEasyVersion-v0',
        entry_point='gym.envs.classic_control.mountain_car:MountainCarEnv',
        max_episode_steps=500,      # MountainCar-v0 uses 200
        reward_threshold=-110.0
    )

def run(render=None, training_cycles=1, num_episodes=1, epsilon=None, epsilon_decay_span=None, epsilon_min=None,
        learning_rate=None, discount_factor=None, replay_init_size=50000,
        load_weights=None, save_weights=None, work_dir=None):

    # env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
    # env = gym.make("ALE/Breakout-v5", obs_type="ram")
    # env = gym.make("ALE/Breakout-v5", render_mode="human")
    best_reward = 0
    total_frames = 0
    timer = Timer()
    best_play_reward = -1000

    register_gym_mods()

    if render == 'human':
        env = gym.make("MountainCarMyEasyVersion-v0", render_mode="human")
    else:
        env = gym.make("MountainCarMyEasyVersion-v0")

    agent = AgentClassicControlDqn(load_weights=load_weights, work_dir=work_dir,
                             epsilon=epsilon, epsilon_decay_span=epsilon_decay_span, epsilon_min=epsilon_min,
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
    del agent

    total_time = timer.cumulative_times["Training cycle"]
    total_mins = int(total_time // 60)
    total_secs = total_time - (total_mins * 60)
    print(f"\nRun finished. Total time taken: {total_mins} mins {total_secs:0.1f} secs")


def main():
    # run(render='human')
    run(
        # render='human',
        epsilon=1.0,
        epsilon_decay_span=25000,
        training_cycles=20,
        num_episodes=10,
        learning_rate=0.001,
        # load_weights="weights.h5",
        replay_init_size=1000
        # work_dir="breakout_temp"
    )
    # run(render='human', training_cycles=5, num_episodes=0, load_weights="batched_updates_2.h5",
    #     replay_init_size=0)
    # run(training_cycles=2, num_episodes=1)


if __name__ == '__main__':
    main()