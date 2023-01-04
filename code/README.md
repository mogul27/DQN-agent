# Source code used during Reinforcement Learning project

To run the code you need to install the Arcade Learning Environment. The following should achieve this:

- pip install ale-py
- pip install autorom
- pip install -U gym
- pip install -U gym[atari,accept-rom-license]

The following code will confirm atari gym is installed OK:

``` python
import random
import gym

env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
obs, info = env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
while not terminated and steps < 2000:
  steps += 1
  action = random.choice([0, 1, 2, 3])
  obs, reward, terminated, truncated, info = env.step(action)
  total_reward += reward

print(f"Game Over : Score {total_reward}, steps={steps}")
```

You need cv2 for the reformatting of the game images returned by the environment:

- pip install opencv-python

And, you will also need the following for the neural networks:

- tensorflow / keras : https://www.tensorflow.org/install
- pytorch : https://pytorch.org/get-started/locally/

## DQN

## Advantage Actor Critic (A2C)

## Asynchronous n-step Q-Learning

Main source code is in:

- atari_asynch_q_learning.py

This has many options which are listed at the bottom of the file. 

The default setting is to run 10 episodes of breakout with 1 worker process and record stats every 3 episodes.
It will generate some files (a png, and some csv data files) in folder : async/breakout

It takes between 3 and 4 hours to run 10,000 episdoes with 8 workers on Google colab. It should be scoring about
30 points a game by then.

Supporting utilities are in:

- dqn_utils.py