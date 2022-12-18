from Breakout_DQN_agent import PreProcessing
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    frame_skip = 20
    fading_rate = 0.85
    env = gym.make('ALE/Breakout-v5', render_mode=None, frameskip=1)
    env = AtariPreprocessing(env,frame_skip=1)
    state, info = env.reset()
    preprocess = PreProcessing()
    env.action_space.sample()
    action = 1
    obs_history = np.zeros((84,84,1))
    for _ in range(frame_skip):
        obs, reward, terminal, truncated, info = env.step(action)
        obs = obs.reshape(84,84,1)
        obs_history = np.concatenate((obs_history,obs), axis=2)
        obs_history = obs_history * fading_rate
    obs_history_final = np.max(obs_history,axis = 2)
    plt.imshow(obs_history_final)
    plt.show()
    