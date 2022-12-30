import gym
import numpy as np
import copy
from disc_networks import ConvActor, ConvCritic
from a2c_utilities import StateHistory
import matplotlib.pyplot as plt

def main(gamma :float=0.99, actor_lr: float=0.001,
         critic_lr: float=0.001, num_actions: int=4,
         num_episodes: int=1000, load_weights: bool=False,
         window_size: int=10) -> None:
    """Run main loop for training A2C network 
    
    Parameters:
    gamma (float): discount factor for temporal learning error (default 0.9)
    actor_lr (float): Learning rate for the Actor neural network (default 0.001)
    critic_lr (float): Learning rate for the Critic neural network (default 0.001)
    num_actions (int): Number of actions (control points) for the agent (default 4)
    num_episodes (int): Number of episodes for which to train the agent (default 1000)
    load_weights (bool): If True, loads stored weights for Actor and Critic 
    networks (default False)
    window_size (int): Number of episodes for which to compute the running performance
    average (default 10)

    Returns:
    None
    """

    # Initialise a new environment
    env = gym.make("BreakoutNoFrameskip-v4")

    # Apply preprocessing from original DQN paper including greyscale and cropping
    wrapped_env = gym.wrappers.AtariPreprocessing(env)
    action_space = [i for i in range(wrapped_env.action_space.n)]

    # Define the input dimensions for networks as observation space with 
    # 4 channels for 4 frames in each network input
    state_dims = wrapped_env.observation_space.shape
    state_dims = state_dims + (4,)

    # Initialise actor and citic + corresponding networks
    actor = ConvActor(entropy_weight=0.06)
    actor.create_network(state_dims, num_actions, actor_lr)
    critic = ConvCritic()
    critic.create_network(state_dims, critic_lr)

    if load_weights:
        actor.load_network_weights(game_type='atari')
        critic.load_network_weights(game_type='atari')

    # Initialise list to hold rewards across episodes
    episode_reward_list = []
    window_average_list = []

    # Episode begins here
    for episode_count in range(1, num_episodes):

        # Initialise variables to track episode progress and performance
        steps = 0
        episode_reward = 0

        # Reset environment and retrieve number of lives
        prev_state, info = wrapped_env.reset()
        prev_lives = info['lives']

        # Initialise a state history which holds 4 previous frames for network input
        state_history = StateHistory()
        # Fill state history with empty states
        state_history.empty_state()

        # Add initial state into StateHistory
        state_history.data.pop(0)
        state_history.data.append(prev_state)
        prev_network_input = state_history.get_states()
        
        terminal = False
        truncated = False

        # fire as first action 
        next_action = 1

        while not terminal and not truncated:
            # Inner loop = 1 step from here
            
            # Take action using Actor
            action = actor.predict(prev_network_input, action_space)

            # Force agent to fire if no steps taken or new life
            if next_action:
                action = 1
                next_action = None
            
            # Step the environment and retrieve key information
            next_state, reward, terminal, truncated, info = wrapped_env.step(action)
            next_lives = info['lives']

            # Add next_state into StateHistory and remove oldest state
            state_history.data.pop(0)
            state_history.data.append(next_state)

            # Define network input from state history after stepping environment
            next_network_input = state_history.get_states()

            # Value current and next state with critic (v(St))
            prev_state_value = critic.predict(prev_network_input)
            next_state_value = critic.predict(next_network_input)

            # Send terminal flag if life lost without terminating episode
            if next_lives < prev_lives and not terminal:
                td_target = reward + 0 * gamma * next_state_value
                next_action = 1
            
            else:
                # Get the td update target value
                # Use 1 - terminal since no future value for terminal states
                td_target = reward + (1-terminal) * gamma * next_state_value

            # Calc extra reward for action at state vs mean reward for action in
            # that state
            adv_function_approx = td_target - prev_state_value
            
            # Update network parameters 
            # Pass td_target to critic
            critic.train(prev_network_input, td_target)
            
            # Actor updates using the td_target adjusted for baseline (advantage) to make it A2C
            actor.train(prev_network_input, adv_function_approx, action, num_actions)

            # Increment steps and set prev_state = next_state
            prev_state = copy.deepcopy(next_state)
            prev_lives = copy.deepcopy(next_lives)
            prev_network_input = copy.deepcopy(next_network_input)
            episode_reward += reward
            steps += 1

        # Add episode reward to list at end of episode
        episode_reward_list.append(episode_reward)

        # Calculate an N episode running average
        if episode_count > window_size:
            window = episode_reward_list[-window_size:]
            running_average = np.mean(window)

        else:
            # If less than 10 steps - set to default value to avoid errors with slicing
            running_average = 1.0

        window_average_list.append(running_average)

        # Output performance sumarry
        print("Episode: {}, Total Reward: {}, Steps: {}, {}-Window_Average: {}"
              .format(episode_count, episode_reward, steps, window_size, running_average))

        # Record performance summaries to provide a stateful store of performance
        if episode_count == 1:
            with open('atarirewards.txt', 'w') as reward_txt:
                reward_txt.write("Episode: {}, Total Reward: {}, Steps: {}, {}-Window_Average: {}"
                .format(episode_count, episode_reward, steps, window_size, running_average))
        else:
            with open('atarirewards.txt', 'a') as reward_txt:
                reward_txt.write("\n Episode: {}, Total Reward: {}, Steps: {}, {}-Window_Average: {}"
                .format(episode_count, episode_reward, steps, window_size, running_average))
    
        # save weights every 100 episodes
        if episode_count % 50 == 0:
            critic.save_network_weights(game_type="atari", episode=episode_count)
            actor.save_network_weights(game_type="atari", episode=episode_count)

        # Every 1000 episodes display a plot of progress
        if episode_count % 2000 == 0:
            episode_axis = [i for i in range(episode_count)]
            plt.plot(episode_axis, window_average_list)
            plt.xlabel("Episode")
            plt.xticks(np.arange(0, episode_count, step=750))
            plt.ylabel("{}-Window Average Reward".format(window_size))
            plt.show()

    
    # After num_episodes of episodes have completed, show performance on every episode
    episode_axis = [i for i in range(num_episodes)]
    plt.plot(episode_axis, episode_reward_list)
    plt.xlabel("Episode")
    plt.xticks(episode_axis)
    plt.ylabel("Episode Reward")
    plt.show()




if __name__ == "__main__":

    main(gamma=0.99, actor_lr=0.0001, critic_lr=0.0001, num_actions=4,
         num_episodes=10000, load_weights=False, window_size=10)







