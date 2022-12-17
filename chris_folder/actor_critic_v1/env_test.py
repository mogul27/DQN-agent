import gym
import numpy as np

env = gym.make("BipedalWalker-v3", hardcore=False)
prev_state, info = env.reset()

num_actions = env.action_space
#possible_actions = [n for n in range(num_actions)]
state_dims = env.observation_space.shape

print(num_actions)
print(state_dims)
# for i in range(100):
#     action = np.random.uniform(-1.0, 1.0, size=4)
#     state, reward, terminal, _, _ = env.step(action)
#     print(state)
#     env.render()