import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
from keras.models import Sequential     # ver 2.9.0
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import time
import pickle
from enum import Enum
from collections import defaultdict


def test_env(env):
    # test the environment 
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, info = env.reset()

#TODO: implement frame skipping
#TODO: read the deepmind papar once more.



class DQN_Agent:
    def __init__(self, eps_schema=1,memory_size = 400000, verbose = 1, save = False):
        self.eps_start  = 1
        self.eps_end = 0.1
        self.annealing_span = 90000
        self.eps = self.eps_start 
        self.eps_schema = eps_schema
        self.gamma = 0.95
        self.SAVE_MODELS = save
        self.verbose = verbose
        self.q_net = self.init_cnn()
        self.target_net = self.init_cnn()
        self.sync_nets()
        self.memory = Memory(N=memory_size, save=save)
        self.steps = 0
        self.step_threshold = 10000
    
    def load_q_net():
        pass

    def save_q_net():
        pass

    def save_target_net():
        pass

    def load_target_net():
        pass
        
    def select_action(self, env, state):
        if self.eps_schema == Eps_Type.ANNEALING:
            self.anneal_eps()
        p = np.random.random()
        if p <= self.eps:
            action = env.action_space.sample()
        else:
            q_vals = self.q_net.predict_on_batch(state)
            action = np.argmax(q_vals)
        return action


    def train(self, env, n_episodes=100):
        preprocess = PreProcessing()
        self.steps = 0
        history = Training_History(n_episodes)
        history.start_time()
        for n_ep in range(n_episodes):
            state, info = env.reset()
            preprocess.reset(info['lives'])
            state = preprocess.preprocess_state(state)
            terminal =  False
            truncated = False
            self.save_nets()
            episode_steps =0
            total_undiscouted_return = 0
            while not terminal and not truncated:
                self.steps += 1
                episode_steps += 1
                action = self.select_action(env, state)

                observation, reward, terminal, truncated, info = env.step(action)
                observation = preprocess.preprocess_state(observation)
                reward = preprocess.preprocess_reward(reward,info)
                total_undiscouted_return += reward

                self.memory.add_experience(state, action, observation, reward, terminal)
                if self.steps > self.step_threshold:
                    self.update_q_net_in_batch(batch_size=16)
                state = observation 
            history.update(n_ep, total_undiscouted_return, episode_steps, self.eps, verbose=1)
            
        self.save_nets()
        self.memory.save()
        
        
    def save_nets(self):
        if self.SAVE_MODELS and self.steps % 20 == 0:
            self.save_q_net() 
            self.save_target_net()
        return


    def anneal_eps(self):
        if self.steps < self.step_threshold:
            eps = self.eps_start
        else:
            eps = self.eps - (self.eps_start - self.eps_end)/self.annealing_span
            if eps < self.eps_end:
                eps = self.eps_end
        self.eps = eps
        return

    def update_q_net_in_batch(self,batch_size=16):
        s, next_s, a, r, not_terminal =  self.memory.get_experiences_in_batch(batch_size)
         # use the target net to compute the update target
        y = r +  not_terminal * self.gamma * np.max(self.target_net.predict_on_batch(next_s), axis=1)
        # use the q net to compute the estimated q values for all actions during the forward pass
        y_hat = self.q_net.predict_on_batch(s)
        # overwirte the value for the taken action with the future expected return calculated by the target net
        y_hat[range(y_hat.shape[0]),a] = y
        # during the backward pass, update the weights of the net to reduce the rmse.
        # NOTE: only the action value for the current action  will contribute to the loss.
        # therefore the weights will be adjusted accordingly
        self.q_net.train_on_batch(s, y_hat) # TODO Instead of fitting the results of each replay independently, I could first simaulte all the rreply then fit the network cosidering all datapoints at once. It should be more effient
        return

    def sync_nets(self):
        w = self.q_net.get_weights()
        self.target_net.set_weights(w)
        return

    def init_cnn(self, learning_rate=0.001):
        cnn = Sequential()
        cnn.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(84,84,1)))
        cnn.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        cnn.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dense(4,activation=None)) # no activation function (means the output is the sum of the input to the neuron)
        cnn.compile(optimizer="adam", loss="mean_squared_error",metrics = "mean_squared_error")
        return cnn


class Eps_Type(Enum):
    CONSTANT = 1
    ANNEALING = 2


class Training_History():
    def __init__(self, n_episodes) -> None:
        self.history = defaultdict(list)
        self.history['score'] = np.zeros(n_episodes)
        self.history['steps'] = np.zeros(n_episodes)
        self.history['time'] = np.zeros(n_episodes)
        self.history['episoln'] = np.zeros(n_episodes)
        self.history['avg_score_10'] = np.zeros(n_episodes)
        self.history['frames_hour'] = np.zeros(n_episodes)
    
    def update(self, ep, score, steps, eps, verbose=0):
        self.history['score'][ep] = score
        self.history['steps'][ep] = steps
        self.history['episoln'][ep] = round(eps,3)
        self.history['time'][ep] = self.delta_time()
        self.history['avg_score_10'][ep] = self.calculate_moving_avg(10,ep)
        self.history['frames_hour'][ep] = round(self.history['steps'][ep] / self.history['time'][ep] * 3600,0)
        if verbose > 0:
            self.print_summary(ep)

    def delta_time(self):
        delta = time.time() - self.start_time
        self.start_time = time.time()
        return delta

    def start_time(self):
        self.start_time = time.time()
    
    def calculate_moving_avg(self,window,ep):
        return round(np.mean(self.history['score'][(ep-window)*(ep>window):ep+1]),2)

    def print_summary(self, ep):
        print(f"Episode: {ep}, "
            f"Total undiscouted reward = {self.history['score'][ep]}, "
            f"Steps = {self.history['steps'][ep]}, "
            f"eps = {self.history['episoln'][ep]}, "
            f"10 games avg score = {self.history['avg_score_10'][ep]}, "
            f"Cumulative steps = {np.sum(self.history['steps'][0:ep+1])}, "
            f"Frames per hour = {np.sum(self.history['frames_hour'][ep])}")

class PreProcessing:
    
    def __init__(self, n_lives=5) -> None:
        self.lives = n_lives

    def preprocess_state(self, s):
        return s.reshape(1,84,84,1)
    
    def preprocess_reward(self, r, info):
        if self.lives > info['lives']:
            r = -1
            self.lives = info['lives']
        return r

    def reset(self,n_lives):
        self.lives = n_lives


class Memory:
    def __init__(self,N=0, save=False):
        self.N = 0
        self.buffer = [() for _ in range(N)]
        self.idx = 0
        self.is_full = False
        self.SAVE = save
    

    def add_experience(self, state, action, observation, reward, terminal):
        # Store observations in buffer.
        if self.idx == self.N:
            if not self.is_full:
                print("Memory buffer reached capacity") 
            self.idx = 0
            self.is_full = True
        self.buffer[self.idx] =  (state, action, observation, reward, int(not terminal))
        self.idx +=1

    def get_experiences_in_batch(self, batch_size = 16):
        batch_idxs =  np.random.randint(0, self.get_upper_bound(), batch_size)
        s = []; a = [];  obs = []; r = []; t = []
        [[s.append(self.buffer[i][0]),
          a.append(self.buffer[i][1]),
          obs.append(self.buffer[i][2]),
          r.append(self.buffer[i][3]),
          t.append(self.buffer[i][4])] for i in np.unique(batch_idxs)]
          
        s_batch = np.concatenate(s,axis=0)
        a_batch = np.array(a)
        obs_batch = np.concatenate(obs,axis=0)
        r_batch = np.array(r)
        t_batch = np.array(t)
        return s_batch, obs_batch, a_batch, r_batch, t_batch

    def get_upper_bound(self):
        return max(self.idx, self.is_full * self.N)

    def save(self, file_name):
        if self.SAVE:
            with open(file_name, 'wb') as file:
                pickle.dump(self.buffer, file)

    def load(self, file_name):
        with open(file_name, 'rb') as file:
            self.buffer = pickle.load(file)

if __name__ == "__main__":
    # Create the environment
    env = gym.make('ALE/Breakout-v5', render_mode=None, frameskip=1)
    nv = AtariPreprocessing(env,frame_skip=4)
    #test_env(env)
    agent = DQN_Agent(eps_schema=Eps_Type.ANNEALING, save=False, verbose=1)
    agent.train(env, n_episodes=4000)