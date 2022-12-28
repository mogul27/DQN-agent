import gym
import numpy as np
import copy
from disc_networks import ConvActor, ConvCritic
from a2c_utilities import StateHistory
import matplotlib.pyplot as plt

def main(gamma :float=0.99, actor_lr: float=0.001,
         critic_lr: float=0.001, num_actions: int=4,
         num_episodes: int=1000) -> None:
    """Run main loop for training A2C network on bipedal walker env
    
    Parameters:
    gamma (float): discount factor for temporal learning error (default 0.9)
    actor_lr (float): Learning rate for the Actor neural network (default 0.001)
    critic_lr (float): Learning rate for the Critic neural network (default 0.001)
    num_actions (int): Number of actions (control points) for the agent (default 4)
    num_episodes (int): Number of episodes for which to train the agent

    Returns:
    None
    """

    # Initialise a new environment
    env = gym.make("BreakoutNoFrameskip-v4")

    load_weights = False
    # Apply preprocessing from original DQN paper including greyscale and cropping
    wrapped_env = gym.wrappers.AtariPreprocessing(env)
    action_space = [i for i in range(env.action_space.n)]



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

    # Initialise list to hold rewards across episodes and parameters for recording performance
    episode_reward_list = []
    window_size = 10
    window_average_list = []

    # Episode begins here
    for episode_count in range(1, num_episodes):

        # Initialise variables to track episode progress and performance
        steps = 0
        episode_reward = 0

        prev_state, info = wrapped_env.reset()
        prev_lives = info['lives']

        state_history = StateHistory()
        # Fill state history with empty states
        state_history.empty_state()

        # Add into StateHistory
        state_history.data.pop(0)
        state_history.data.append((prev_state, False))
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

            next_state, reward, terminal, truncated, info = wrapped_env.step(action)
            next_lives = info['lives']

            # Add next_state into StateHistory and remove oldest state
            state_history.data.pop(0)
            state_history.data.append((next_state, terminal))

            next_network_input = state_history.get_states()

            # Value current and next state with critic (v(St))
            prev_state_value = critic.predict(prev_network_input)
            next_state_value = critic.predict(next_network_input)

            # send terminal flag if life lost without terminating episode
            if next_lives < prev_lives and not terminal:
                td_target = reward + 0 * gamma * next_state_value
                next_action = 1
            
            else:
                # Get the td update target value (Week 3 of content)
                # Use 1 - terminal since no future value for terminal states
                td_target = reward + (1-terminal) * gamma * next_state_value

            # Calc extra reward for action at state vs mean reward for action in
            # that state which is the same as a full TD update to existing value
            # See lecture material Week 3

            adv_function_approx = td_target - prev_state_value
            #print(action)
            #print(adv_function_approx)

            # Update network parameters 

            # Critic gets the td_target as the loss is this squared (Uses rmse)
            # as in TD learning (Lectures week 3)
            critic.train(prev_network_input, td_target)
            
            # Actor updates using the Q value adjusted for advantage to make it A2C
            # Negative lo loss of sample distribution calclated in train function
            actor.train(prev_network_input, adv_function_approx, action, num_actions)

            # Increment steps and set prev_state = next_state
            prev_state = copy.deepcopy(next_state)
            prev_lives = copy.deepcopy(next_lives)
            prev_network_input = copy.deepcopy(next_network_input)
            episode_reward += reward
            steps += 1

        episode_reward_list.append(episode_reward)
        # Calculate an N episode running average
        
        if episode_count > window_size:
            window = episode_reward_list[-window_size:]
            running_average = np.mean(window)

        else:
            running_average = episode_reward

        window_average_list.append(running_average)

        print("Episode: {}, Total Reward: {}, Steps: {}, {}-Window_Average: {}"
              .format(episode_count, episode_reward, steps, window_size, running_average))
        # Record

        if episode_count == 1:
            with open('atarirewards.txt', 'w') as reward_txt:
                reward_txt.write("Episode: {}, Total Reward: {}, Steps: {}, {}-Window_Average: {}"
                .format(episode_count, episode_reward, steps, window_size, running_average))
        else:
            with open('atarirewards.txt', 'a') as reward_txt:
                reward_txt.write("\n Episode: {}, Total Reward: {}, Steps: {}, {}-Window_Average: {}"
                .format(episode_count, episode_reward, steps, window_size, running_average))
    
        # save weights every 100 episodes
        if episode_count % 100 == 0:
            critic.save_network_weights(game_type="atari", episode=episode_count)
            actor.save_network_weights(game_type="atari", episode=episode_count)

        # Every 1000 episodes display a plot of progress
        if episode_count % 14 == 0:
            episode_axis = [i for i in range(episode_count)]
            plt.plot(episode_axis, window_average_list)
            plt.xlabel("Episode")
            plt.xticks(np.arange(0, episode_count, step=200))
            plt.ylabel("{}-Window Average Reward".format(window_size))
            plt.show()

    
    # After num_episodes of episodes have run
    episode_axis = [i for i in range(num_episodes)]
    plt.plot(episode_axis, episode_reward_list)
    plt.xlabel("Episode")
    plt.xticks(episode_axis)
    plt.ylabel("Episode Reward")
    plt.show()




if __name__ == "__main__":

    main(gamma=0.99, actor_lr=0.0001, critic_lr=0.0001, num_actions=4,
         num_episodes=10000)







