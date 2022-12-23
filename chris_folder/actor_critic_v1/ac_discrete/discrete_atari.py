import gym
import numpy as np
import copy
from disc_networks import ConvActor, ConvCritic
import matplotlib.pyplot as plt

def main(gamma :float=0.99, actor_lr: float=0.001,
         critic_lr: float=0.001, num_actions: int=2,
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

    # Create Environment and get environment attributes
        # Initialise a new environment
    env = gym.make("BreakoutNoFrameskip-v4", frameskip=1)

    # Apply preprocessing from original DQN paper including greyscale and cropping
    wrapped_env = gym.wrappers.AtariPreprocessing(env)
    action_space = [i for i in range(env.action_space.n)]
    prev_state, info = env.reset()


    state_dims = wrapped_env.observation_space.shape
    state_dims = state_dims + (1,)

    # Initialise actor and citic + corresponding networks
    actor = ConvActor()
    actor.create_network(state_dims, num_actions, actor_lr)
    critic = ConvCritic()
    critic.create_network(state_dims, critic_lr)

    # Initialise list to hold rewards across episodes
    episode_reward_list = []

    # Episode begins here
    for episode_count in range(num_episodes):

        # Initialise variables to track episode progress and performance
        steps = 0
        episode_reward = 0
        actions = []

        prev_state, info = wrapped_env.reset()
        prev_lives = info['lives']
        
        terminal = False
        truncated = False

        while not terminal and not truncated:
            # Inner loop = 1 step from here
            
            # Take action using Actor
            action = actor.predict(prev_state, action_space)
            actions.append(action)

            # # Force agent to fire if no steps taken
            if steps > 20:
                if 1 not in actions:
                    action = 1

            next_state, reward, terminal, truncated, info = wrapped_env.step(action)
            next_lives = info['lives']

            # Value current and next state with critic (v(St))
            prev_state_value = critic.predict(prev_state)
            next_state_value = critic.predict(next_state)

            # send terminal flag if life lost without terminating episode
            if next_lives < prev_lives and not terminal:
                td_target = reward + 0 * gamma * next_state_value
            
            else:
                # Get the td update target value (Week 3 of content)
                # Use 1 - terminal since no future value for terminal states
                td_target = reward + (1-terminal) * gamma * next_state_value

            # Calc extra reward for action at state vs mean reward for action in
            # that state which is the same as a full TD update to existing value
            # See lecture material Week 3

            adv_function_approx = td_target - prev_state_value
            print(action)
            print(adv_function_approx)

            # Update network parameters 

            # Critic gets the td_target as the loss is this squared (Uses rmse)
            # as in TD learning (Lectures week 3)
            critic.train(prev_state, td_target)
            
            # Actor updates using the Q value adjusted for advantage to make it A2C
            # Negative lo loss of sample distribution calclated in train function
            actor.train(prev_state, adv_function_approx, action, num_actions)

            # Increment steps and set prev_state = next_state
            prev_state = copy.deepcopy(next_state)
            prev_lives = copy.deepcopy(next_lives)
            episode_reward += reward
            steps += 1
            
        episode_reward_list.append(episode_reward)
        print("Episode: {}, Total Reward: {}, Steps: {}"
              .format(episode_count, episode_reward, steps))
        # Record

        if episode_count == 0:
            with open('atarirewards.txt', 'w') as reward_txt:
                reward_txt.write("Episode: {}, Total Reward: {}, Steps: {}".format(
                    episode_count, episode_reward, steps))
        else:
            with open('atarirewards.txt', 'a') as reward_txt:
                reward_txt.write("\n Episode: {}, Total Reward: {}, Steps: {}".format(
                    episode_count, episode_reward, steps))
    
        # save weights every 100 episodes
        if episode_count % 100 == 0:
            critic.save_network_weights(game_type="atari")
            actor.save_network_weights(game_type="atari")
    
    # After num_episodes of episodes have run
    episode_axis = [i for i in range(num_episodes)]
    plt.plot(episode_axis, episode_reward_list)
    plt.xlabel("Episode")
    plt.xticks(episode_axis)
    plt.ylabel("Episode Reward")
    plt.show()




if __name__ == "__main__":

    main(gamma=0.9, actor_lr=0.0001, critic_lr=0.0001, num_actions=4,
         num_episodes=1000)







