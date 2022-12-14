import gym
import numpy as np
import copy
from dqn_agent import DQNAgent
from dqn_utilities import StateHistory
from helper_functions import fill_replay_buffer

import sys
np.set_printoptions(threshold=sys.maxsize)

def main(load_weights: bool=False, minibatch_size:int=32, experience_buffer_size: int=100000
        ,max_episodes: int=1000, starting_epsilon: int=1, epsilon_decay: float=9e-6, final_epsilon: int=0.1
        ,gamma: float=0.99, update_steps: int=1000, game: str="ALE/Breakout-v5"):

    # Set algorithm parameters
    minibatch_size = minibatch_size
    experience_buffer_size = experience_buffer_size
    max_episodes = max_episodes
    epsilon = starting_epsilon
    epsilon_decay = epsilon_decay
    final_epsilon = final_epsilon
    gamma = gamma
    c = update_steps # How many steps until update parameters of networks to match

    # Initialise a new environment
    env = gym.make(game, frameskip=1)

    # Apply preprocessing from original DQN paper including greyscale and cropping
    wrapped_env = gym.wrappers.AtariPreprocessing(env)

    # Initialise history object for initial wandering
    random_history = StateHistory()
    random_history.empty_state()

    prev_state, info = wrapped_env.reset()

    # Append prev_state to random_history
    random_history.data.pop(0)
    random_history.data.append((prev_state, False))

    # Collect info about environment and actions for constructing network outputs
    num_actions = wrapped_env.action_space.n
    possible_actions = [n for n in range(num_actions)]
    state_dims = wrapped_env.observation_space.shape

    # Concatenate 4 to state dims to represent the number of channels
    # which is 4 because 4 frames of greyscale images used
    state_dims = state_dims + (4,)
    
    # Initialise a new DQNAgent
    agent = DQNAgent(epsilon=epsilon, max_buffer_len=experience_buffer_size, gamma=gamma)

    if load_weights:
        agent.load_weights()
    else:
        agent.initialise_network_buffer(num_actions, state_dims)

    # Fill the agent's replay buffer with random wandering for initial experience
    fill_replay_buffer(experience_buffer_size=experience_buffer_size, 
                       possible_actions=possible_actions,
                       random_history=random_history,
                       agent=agent,
                       wrapped_env=wrapped_env)

    # Initialise episode monitoring (rewards, num_episodes)
    total_episode_rewards= []
    episode_counter = 0
    step_number = []

    # Use all_steps to track when to update target network across episodes
    all_steps = 0

    for episode_counter in range(max_episodes):

        # Reset environment now that replay buffer filled or new episode started
        env = gym.make(game, frameskip=1)
        state_history = StateHistory()
        # Fill state history with empty states
        state_history.empty_state()

        wrapped_env = gym.wrappers.AtariPreprocessing(env)
        prev_state, info  = wrapped_env.reset()

        state_history.data.pop(0)
        state_history.data.append((prev_state, False))

        # Get lives at previous step
        prev_lives = info['lives']

        # Set terminal and truncated for loop
        terminal = False
        truncated = False

        step=0
        rewards=[]
        # Collect non-adjusted rewards
        real_rewards = []

        # Perform an episode of training
        while not terminal and not truncated:
            
            network_input = state_history.get_states()
            action = agent.epsilon_greedy_selection(num_actions,
                                                    possible_actions, network_input)
            next_state, reward, terminal, truncated, info = wrapped_env.step(action)
            real_rewards.append(reward)
            # Adjust rewards if lives lost and set prev_lives = lives
            # reward, prev_lives = agent.adjust_reward_lives(reward, info, prev_lives)
            rewards.append(reward)

            # Create a copy of the state history object so it is not mutated in memory
            # then put the next state into the original object
            buffer_history = copy.deepcopy(state_history)
            state_history.data.pop(0)
            state_history.data.append((next_state, terminal))
            # Make a copy of next_state history object to store in memory to avoid mutating it
            next_buffer_history = copy.deepcopy(state_history)

            agent.experience_buffer.add(buffer_history, action, reward, next_buffer_history)
            prev_state = copy.deepcopy(next_state)

            minibatch = agent.experience_buffer.get_random_data(minibatch_size)

            network_input_batch = []
            target_preds_batch = []

            for experience in minibatch:
                
                # Unpack experience
                # Label with exp to avoid overwriting current state
                prev_history_exp, action_exp, reward_exp, next_buffer_history_exp = experience

                # Reshape next_state to be passed into network
                network_input = next_buffer_history_exp.get_states()
                
                # Retrieve q2 value for the best action
                best_action_val = agent.get_q2_preds(network_input, all_steps)

                # Retrieve q1 predictions which function as y_hat
                target_preds = agent.get_q1_action_values(network_input)
                
                if next_buffer_history_exp.is_terminal():
                    target_preds[action_exp] = reward_exp
                else:
                    # Retrieve best possible action according to target network
                    target_preds[action_exp] = reward_exp + gamma*best_action_val

                # Reshape prev_state to be passed into network
                network_input = prev_history_exp.get_states()

                # use [0] to avoid batch_size being added as an extra_dim
                network_input_batch.append(network_input[0])
                target_preds_batch.append(target_preds)

            network_input_batch = np.array(network_input_batch)
            target_preds_batch=np.array(target_preds_batch)
            agent.q1.train_on_batch(network_input_batch, target_preds_batch)
            
            # Every 200 timesteps
            if step % 200 == 0:
                print("Step: {}, Epsilon: {}".format(step, epsilon))

            # Every c timesteps
            if all_steps % c == 0:
                print("Network Updated")
                agent.q2.set_weights(agent.q1.get_weights())
            
            step += 1
            all_steps += 1
            
            if epsilon > final_epsilon:
                epsilon = epsilon - epsilon_decay
                # If decay takes it under 0.1
            if epsilon < final_epsilon:
                epsilon = final_epsilon

        total_episode_rewards.append(sum(real_rewards))
        step_number.append(step)
        print("episode {} complete. Total episode Reward: {}".format(episode_counter, sum(real_rewards)))

        # Save weights and write episode reward
        agent.save_weights()

        if episode_counter == 1:
            with open('rewards.txt', 'w') as reward_txt:
                reward_txt.write("Episode: {}, Total Reward: {}, Steps: {}".format(
                                episode_counter, sum(real_rewards), step))
        else:
            with open("rewards.txt", "a") as reward_txt:
                # Append next epsiode reward at the end of file
                reward_txt.write("\nEpisode: {}, Total Reward: {}, Steps: {}".format(
                                episode_counter, sum(real_rewards), step))

    print("Episode total rewards: ", total_episode_rewards)

if __name__ == "__main__":

    # Override values
    #experience_buffer_size = 32
    epsilon_decay=9e-5

    main(load_weights=False, minibatch_size=32, experience_buffer_size=100000, max_episodes=1000, 
    starting_epsilon=1, epsilon_decay=9e-6, final_epsilon=0.1, gamma=0.99, update_steps=500
    , game="ALE/Breakout-v5")