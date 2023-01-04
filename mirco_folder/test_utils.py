from Breakout_DQN_agent import DQN_Agent, Eps_Type
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle




def test_agent(env, agent, net_name):
    agent.test_eps = 0
    agent.load_q_net("./"+net_name)
    avg, scores = agent.play(env, n_episodes=30, frame_skip=4)
    return scores


def test(env,folder_name):
    agent = DQN_Agent(eps_schema=Eps_Type.ANNEALING_LINEAR, memory_size=100000, save=False, verbose=1)
    net_list = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and ".h5" in f]
    net_scores = defaultdict(list)
    n_nets  = len(net_list)
    for i, net in enumerate(net_list):
        print(f" {i}/{n_nets}. Testing {net}.")
        net_score = test_agent(env, agent, net)
        net_scores[net] = net_score
    return net_scores


def test_agent_performances():
    env = gym.make('ALE/Breakout-v5', render_mode=None)
    agent = DQN_Agent(eps_schema=Eps_Type.ANNEALING_LINEAR, memory_size=100000, save=False, verbose=1)
    score_dict = test(env, "./")

    with open('./mirco_folder/DQN9/score_test_latest.pickle', 'wb') as file:
       pickle.dump(score_dict, file)
    print("COMPLETED")


def watch_agent_play(net_name, n_episodes=10):
    env = gym.make('ALE/Breakout-v5', render_mode="human",frameskip=1)
    agent = DQN_Agent(eps_schema=Eps_Type.ANNEALING_LINEAR, memory_size=100000, save=False, verbose=1)
    agent.load_q_net("./"+ net_name)
    agent.test_eps = 0
    _, _= agent.play(env, n_episodes=n_episodes, frame_skip=4)


if __name__ == "__main__":
    watch_agent_play("./mirco_folder/DQN9/q_net_83500.h5")