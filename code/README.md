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

Main source code is in:

- DQN_agent.py

All necessary classes and methods to run the DQN agent are within this file.
The file is set up to load an existing trained model (q_net_83500.h5 included in the submission) and watch the agent play the game of Breakout.
To train the agent from scratch, uncomment the correspondent code at the bottom of the file.
On Google Colab with standard GPUs, this implementation should average 250k frames/hour. Therefore, it should take around 8 hours to train the agent for 2M frames. At the 2M frames mark the agent should consistently hit 20 to 30 points per game.
Defult settings intialize a memory buffer capacity of 1M experiences. To reduce RAM consumption, the memory buffer capacity can be reduced by setting a lower value of the input argument `memory_size` in the DQN_Agent constructor call.
The actual agent training starts after 50000 random steps. To commence the training earlier, the memeber `step_threshold` of the class `DQN_Agent` can be set accordingly.



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