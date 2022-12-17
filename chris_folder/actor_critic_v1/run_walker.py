import gym
import numpy as np
import copy
from networks import Actor, Critic

def main(gamma :float=0.9, actor_lr: float=0.001,
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

    # Create Environment and get environment attributes
    env = gym.make("BipedalWalker-v3", hardcore=False)
    state_dims = env.observation_space.shape

    # Initialise actor and citic + corresponding networks
    actor = Actor()
    actor.create_network(state_dims, num_actions, actor_lr)
    critic = Critic()
    critic.create_network(state_dims, critic_lr)

    # Initialise list to hold rewards across episodes
    episode_reward_list = []

    # Episode begins here
    for episode in range(num_episodes):

        # Initialise variables to track episode progress and performance
        steps = 0
        episode_reward = 0

        prev_state, info = env.reset()
        terminal = False
        truncated = False

        while not terminal and not truncated:
            
            # Take action using Actor
            continuous_actions = actor.predict(prev_state)
            next_state, reward, terminal, truncated, info = env.step(continuous_actions)

            # Value current and next state with critic (v(St))
            prev_state_value = critic.predict(prev_state)
            next_state_value = critic.predict(next_state)
            
            # Get the td update target value (Week 3 of content)
            # Use 1 - terminal since no future value for terminal states
            td_target = reward + (1-terminal) * gamma * next_state_value

            # Calc extra reward for action at state vs mean reward for action in
            # that state which is the same as a full TD update to existing value
            # See lecture material Week 3
            adv_function = td_target - prev_state_value
            
            # Update network parameters
            # Actor gets adv_function as an evaluation of its actions 
            # It updates using the Q value adjusted for advantage to make it A2C
            actor.train(prev_state, adv_function)

            # Critic gets the td_target so it updates its estimate of the state value
            # as in TD learning (Lectures week 3)
            critic.train(prev_state, td_target)

            prev_state = copy.deepcopy(prev_state)
            episode_reward += reward
            steps += 1


if __name__ == "__main__":

    main(gamma=0.9, actor_lr=0.001, critic_lr=0.001, num_actions=4,
         num_episodes=1000)







