import gym
import numpy as np

env = gym.make("ALE/DemonAttack-v5", render_mode='human')
env.reset()

action_space_list = [n for n in range(6)]
print(action_space_list)

action_space_size = env.action_space
print("Size:", action_space_size)

for i in range(100):
    action = np.random.choice(action_space_list)
    print(action)
    state, reward, terminal, _, _ = env.step(action)
    env.render()