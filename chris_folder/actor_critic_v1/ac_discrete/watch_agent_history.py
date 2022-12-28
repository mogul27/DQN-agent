import gym
import numpy as np
import copy
from disc_networks import ConvActor, ConvCritic
from a2c_utilities import StateHistory
import matplotlib.pyplot as plt

# Hyperparam placeholders
actor_lr=0.001
critic_lr=0.001
num_actions=4
num_episodes=10

# Create Environment and get environment attributes
# Initialise a new environment
env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')

# Apply preprocessing from original DQN paper including greyscale and cropping
wrapped_env = gym.wrappers.AtariPreprocessing(env)
action_space = [i for i in range(wrapped_env.action_space.n)]
prev_state, info = env.reset()

state_dims = wrapped_env.observation_space.shape
state_dims = state_dims + (4,)

# Initialise actor and citic + corresponding networks
actor = ConvActor()
actor.create_network(state_dims, num_actions, actor_lr)
critic = ConvCritic()
critic.create_network(state_dims, critic_lr)


load_weights = True
if load_weights:
    actor.load_network_weights(game_type='atari')
    critic.load_network_weights(game_type='atari')

# Episode begins here
for episode_count in range(num_episodes):

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
        action = actor.test_predict(prev_network_input, action_space)

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

        # # Value current and next state with critic (v(St))
        # prev_state_value = critic.predict(prev_network_input)
        # next_state_value = critic.predict(next_state)

        # send terminal flag if life lost without terminating episode
        if next_lives < prev_lives and not terminal:
            next_action = 1

        # Increment steps and set prev_state = next_state
        prev_state = copy.deepcopy(next_state)
        prev_lives = copy.deepcopy(next_lives)
        prev_network_input = copy.deepcopy(next_network_input)