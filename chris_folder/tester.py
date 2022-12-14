#import gym
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


test_array = np.array([[[1,2], [3,4]], [[4,5], [5,6]], [[7,8], [9,10]], [[11,12], [13, 14]]])
print(test_array.shape)
reshaped = test_array.reshape(2, 2, 4)
transposed = test_array.transpose()
stack = np.stack(test_array)
st = stack.transpose()
print("Reshaped: \n{}".format(reshaped))
print("Transposed: \n{}".format(transposed))
print("st: \n{}".format(st))

#print(test_array.transpose().shape)
# env = gym.make("ALE/Breakout-v5", render_mode='human', frameskip=4)
# env.reset()

# action_space_list = [n for n in range(4)]
# print(action_space_list)

# action_space_size = env.action_space
# print("Size:", action_space_size)

# for i in range(100):
#     action = np.random.choice(action_space_list)
#     state, reward, terminal, _, _ = env.step(action)
#     print(state)
#     env.render()