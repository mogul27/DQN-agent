import gym
import numpy as np
import copy
from disc_networks import ConvActor, ConvCritic
from a2c_utilities import StateHistory
import matplotlib.pyplot as plt

def main(num_actions: int=4, num_episodes: int=10, load_weights: bool=True,
         sample: bool=True) -> None:
    """Run main loop for viewing A2C performance on rendered environment
    
    Parameters:
    num_actions (int): Number of actions (control points) for the agent (default 4)
    num_episodes (int): Number of episodes for which to watch the agent (default 10)
    load_weights (bool): If True, loads stored weights for Actor and Critic 
    networks (default True)
    sample (bool): If true, actor samples from action probability distribution. Otherwise,
    actor takes the action with the highest probability

    Returns:
    None

    """


    # Initialise a new environment with human rendering
    env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')

    # Apply preprocessing from original DQN paper including greyscale and cropping
    wrapped_env = gym.wrappers.AtariPreprocessing(env)
    action_space = [i for i in range(wrapped_env.action_space.n)]

    # Obtain the expected input dimensions for CNNs
    state_dims = wrapped_env.observation_space.shape
    state_dims = state_dims + (4,)

    # Initialise actor and citic + corresponding networks
    actor = ConvActor()
    actor.create_network(state_dims, num_actions)
    critic = ConvCritic()
    critic.create_network(state_dims)

    # Load previously trained weights
    if load_weights:
        actor.load_network_weights(weights_path_override="GoldenRunWeights/atari_actor_weights3750.h5")
        critic.load_network_weights(weights_path_override="GoldenRunWeights/atari_critic_weights3750.h5")

    # Hold episode rewards
    episode_rewards = []

    # Episode begins here
    for episode_count in range(num_episodes):

        # Initialise episode reward
        episode_reward = 0

        # Reset environment
        prev_state, info = wrapped_env.reset()
        prev_lives = info['lives']

        # Initialise state history to contain current and previous frames
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
            action = actor.test_predict(prev_network_input, action_space, sample=sample)

            # Force agent to fire if no steps taken or new life
            if next_action:
                action = 1
                next_action = None

            # Step environment and obtain relevant informationn
            next_state, reward, terminal, truncated, info = wrapped_env.step(action)
            next_lives = info['lives']

            episode_reward += reward

            # Add next_state into StateHistory and remove oldest state
            state_history.data.pop(0)
            state_history.data.append(next_state)

            next_network_input = state_history.get_states()

            # send terminal flag if life lost without terminating episode
            if next_lives < prev_lives and not terminal:
                next_action = 1

            # Increment steps and set prev_state = next_state
            prev_state = copy.deepcopy(next_state)
            prev_lives = copy.deepcopy(next_lives)
            prev_network_input = copy.deepcopy(next_network_input)

        episode_rewards.append(episode_reward)
        print("Episode Reward: {}".format(episode_reward))
    
    print("Average Episode Reward Over {} Episodes: {}".format(num_episodes, np.mean(episode_rewards)))

if __name__ == "__main__":

    main(num_actions=4, num_episodes=5, load_weights=True, sample=False)