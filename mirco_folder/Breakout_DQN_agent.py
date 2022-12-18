import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
import tensorflow as tf
from keras.losses import Huber
from keras.optimizers import Adam
from keras.models import Sequential, load_model     # ver 2.9.0
from keras.layers import Dense, Conv2D, Flatten, Input
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
    def __init__(self, eps_schema=3 ,memory_size = 200000,  verbose = 1, save = False,  folder="./"):
        self.eps_schema = eps_schema
        # for step_annealing schema
        self.eps_levels =  np.array([1, 0.5,  0.5, 0.3, 0.3,  0.1])
        self.steps_levels = np.array([0, 200000, 250000, 330000, 380000,  500000])
        # for linear_annealing schema
        self.eps_start  = 1
        self.eps_end = 0.1
        self.annealing_span = 700000
        if self.eps_schema==1 or self.eps_schema==2:
            self.eps = self.eps_start
        else:
            self.eps = self.eps_levels[0] 

        self.gamma = 0.99
        self.SAVE_MODELS = save
        self.verbose = verbose
        self.steps = 0
        self.len_states = 4
        self.q_net = self.init_cnn(state_len=self.len_states)
        self.target_net = self.init_cnn(state_len=self.len_states)
        self.sync_steps = 1000
        self.memory = Memory(N=memory_size, save=save)
        self.step_threshold = 30000
        self.history = []
        self.save_folder = folder
        self.test_eps = 0.05
        self.save_each_n_episodes = 100
        self.eval_policy_each_n_episodes = 10
        self.optimizer = Adam(learning_rate=0.00025, clipnorm =1)
        self.loss_function = Huber()
    
    
    def load_q_net(self, file_name):
        self.q_net = load_model(file_name)
      
    def load_target_net(self, file_name):
        self.target_net = load_model(file_name)
      
    def load_memory(self,filename):
        self.memory.load(filename)

    def save_nets(self, ep=0, force=False):
        if (self.SAVE_MODELS and ep % self.save_each_n_episodes == 0) or (self.SAVE_MODELS and force):
            self.save_q_net(ep) 
            self.save_target_net(ep)
        return
    
    def save_nets_and_history(self, ep=0, force=False):
        if (self.SAVE_MODELS and ep % self.save_each_n_episodes == 0) or (self.SAVE_MODELS and force):
            self.memory.save(self.save_folder + "Memory.pickle")
            self.history.save(self.save_folder + "History.pickle")
            self.save_q_net(ep) 
            self.save_target_net(ep)
        return

    def save_q_net(self,ep=0):
        model_name = self.save_folder + "q_net_" + str(ep) + ".h5"
        self.q_net.save(model_name)

    def save_target_net(self,ep=0):
        model_name = self.save_folder + "target_net_" + str(ep) + ".h5"
        self.target_net.save(model_name)

    def calculate_exploration_factor(self):
        if self.eps_schema == Eps_Type.ANNEALING_LINEAR :
            self.linear_anneal_eps()
        elif self.eps_schema == Eps_Type.ANNEALING_STEPS:
            self.stepwise_anneal_eps()

    def select_action(self, env, state, eps=0.05):
        p = np.random.random()
        if p <= eps:
            action = env.action_space.sample()
        else:
            q_vals = self.q_net.predict_on_batch(state)
            action = np.argmax(q_vals)
        return action

    def resume_training(self, env, n_episodes=100, frame_skip = 1, steps=0, eps=1):
        self.steps = steps
        self.eps = eps
        self.train(env, n_episodes,frame_skip)
        return

    def train(self, env, n_episodes=100, frame_skip = 1, batch_size=32):
        preprocess = PreProcessing()
        self.history = Training_History(n_episodes)
        self.history.start_time()
        #self.sync_nets()
        test_score = 0
        for n_ep in range(n_episodes):
            if (self.steps > self.step_threshold) and (n_ep % self.eval_policy_each_n_episodes == 0):
                test_score = self.play(env, 10, 4)
            state, info = env.reset()
            preprocess.reset(info['lives'])
            state = preprocess.preprocess_state(state)
            state = np.concatenate([state, np.zeros((1,84,84,self.len_states-1),dtype=np.uint8)], axis=3)
            terminal =  False
            truncated = False
            self.save_nets_and_history(n_ep)
            episode_steps =0
            total_undiscouted_return = 0
            game_score = 0
            
            while not terminal and not truncated:
                self.steps += 1
                episode_steps += 1
                self.calculate_exploration_factor()
                action = self.select_action(env, state, self.eps)
                self.sync_nets()
                for _ in range(frame_skip):
                    observation, r, terminal, truncated, info = env.step(action)
                    #preprocess.preprocess_terminal(terminal, info)
                    preprocess.preprocess_reward(r,info)
                    #preprocess.overimporse_states(observation, 0.9)
                    game_score += r
                observation = preprocess.preprocess_state(observation)
                #observation =  preprocess.get_overimposed_state()
                reward = preprocess.get_reward()
                #teminal_processed = preprocess.get_terminal()
                total_undiscouted_return += reward

                self.memory.add_experience(state[:,:,:,0].reshape(1,84,84,1), action, observation, reward, terminal)
                loss = np.inf

                if self.steps > self.step_threshold:
                    loss = self.update_q_net_in_batch_GradientTape(batch_size)
                past_state,_,_,_,_ = self.memory.get_experiences_in_batch(batch_size=1, batch_idxs= np.array([self.memory.idx-1]), len_states = self.len_states-1) #batch_size=16, batch_idxs=-1 ,len_states=4
                state = np.concatenate([observation, past_state], axis=3) 
            self.history.update(n_ep, game_score, test_score, episode_steps, self.eps, loss, verbose=1)
        self.save_nets_and_history(n_episodes,force=True)


    def play(self, env, n_episodes=10, frame_skip=4, verbose=0):
        preprocess = PreProcessing()
        sum_game_score = 0
        for _ in range(n_episodes):
            state, info = env.reset()
            state = preprocess.preprocess_state(state)
            state = np.concatenate([state, np.zeros((1,84,84,self.len_states-1),dtype=np.uint8)], axis=3)
            terminal = False
            game_start = True
            noop = 0
            lives = 5
            while not terminal:
                action = self.select_action(env, state, self.test_eps)
                if game_start and action in [0,2,3]:
                  noop +=1
                if noop >= 20:
                  action = 1 #force fire if the agent select 30 action without firing at the beginning of the game
                  noop = 0
                if game_start and action ==1:
                  game_start = False
                for _ in range(frame_skip):
                    observation, r, terminal, truncated, info = env.step(action)
                    #preprocess.overimporse_states(observation, 0.9)
                    sum_game_score += r
                #observation = preprocess.get_overimposed_state()
                if lives > info['lives']:
                    game_start = True
                    lives = info['lives']
                observation = preprocess.preprocess_state(observation)
                state = np.concatenate([observation, state[0,:,:,0:-1].reshape(1,84,84,self.len_states-1)], axis=3)
        avg_score = sum_game_score/n_episodes
        return avg_score
        
    
    def play2(self, env, n_episodes=100):
        frame_skip = 4
        preprocess = [PreProcessing() for _ in range(frame_skip)]
        self.eps_schema = Eps_Type.CONSTANT
        self.eps = 0
        for _ in range(n_episodes):
            state, info = env.reset()
            state = preprocess.preprocess_state(state)

            terminal = False
            while not terminal:
                for k in range(frame_skip):
                    action = self.select_action(env, state)
                    observation, r, terminal, truncated, info = env.step(action)

                    preprocess.overimporse_states(observation, 0.8)
                    #game_score += r
                state = preprocess.get_overimposed_state()
        

    def linear_anneal_eps(self):
        if self.steps < self.step_threshold:
            eps = self.eps_start
        else:
            eps = self.eps - (self.eps_start - self.eps_end)/self.annealing_span
            if eps < self.eps_end:
                eps = self.eps_end
        self.eps = eps
        return
    

    def stepwise_anneal_eps(self):
        if self.steps < self.step_threshold:
            eps = self.eps_levels[0]
        else:
            steps = self.steps - self.step_threshold
            if steps >= self.steps_levels[-1]:
                eps = self.eps_levels[-1]
            else:
                k = np.where(self.steps_levels <= steps)[0][-1]
                m = (self.eps_levels[k+1] - self.eps_levels[k])/ (self.steps_levels[k+1] - self.steps_levels[k])
                eps = self.eps + m
        self.eps = eps
        return


    def update_q_net_in_batch(self,batch_size): 
        s, next_s, a, r, not_terminal =  self.memory.get_experiences_in_batch(batch_size=batch_size, len_states=self.len_states) #  batch_size=16, batch_idxs=[] ,len_states=4
         # use the target net to compute the update target
        next_s = np.concatenate([next_s, s[:,:,:,0:-1]],axis = 3)
        y = r +  not_terminal * self.gamma * np.max(self.target_net.predict_on_batch(next_s), axis=1)
        # use the q net to compute the estimated q values for all actions during the forward pass
        y_hat = self.q_net.predict_on_batch(s) 
        # overwirte the value for the taken action with the future expected return calculated by the target net
        y_hat[range(batch_size),a] = y
        # during the backward pass, update the weights of the net to reduce the rmse.
        # NOTE: only the action value for the current action  will contribute to the loss.
        # therefore the weights will be adjusted accordingly#
        #w = self.q_net.get_weights()
        loss = self.q_net.train_on_batch(s, y_hat)
        #w_new = self.q_net.get_weights()
        return loss
    
    def update_q_net_in_batch_GradientTape(self,batch_size): 
        s, next_s, a, r, not_terminal =  self.memory.get_experiences_in_batch(batch_size=batch_size, len_states=self.len_states) #  batch_size=16, batch_idxs=[] ,len_states=4
         # use the target net to compute the update target
        next_s = np.concatenate([next_s, s[:,:,:,0:-1]],axis = 3)
        y = r +  not_terminal * self.gamma * np.max(self.target_net.predict_on_batch(next_s), axis=1)
        # use the q net to compute the estimated q values for all actions during the forward pass
        mask = tf.one_hot(a,4)
        with tf.GradientTape() as tape:
            q_vals = self.q_net(s)  # need to use model(state) as this returns a tensor which is necessary for automatic differentiation. model.predict returns a numpy array.
            y_hat = tf.reduce_sum(tf.multiply(mask, q_vals), axis =1) 
            loss = self.loss_function(y,y_hat)
        # apply backporop
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.q_net.trainable_variables))
        return loss.numpy()
        #w_new = self.q_net.get_weights()

    def sync_nets(self):
        if (self.steps % self.sync_steps) == 0:
            w = self.q_net.get_weights()
            self.target_net.set_weights(w)
            if self.verbose > 0:
                print('Target network synched')
        return

    def init_cnn(self, adam_learning_rate=0.00025, state_len = 4):
        input_layer = Input(shape = (84,84,4))
        cv1_layer = Conv2D(32, 8, strides=4, activation='relu')(input_layer)
        cv2_layer = Conv2D(64, 4, strides=2, activation='relu')(cv1_layer)
        cv3_layer = Conv2D(64, 3, strides=2, activation='relu')(cv2_layer)
        flatten_layer = Flatten()(cv3_layer)
        dense1_layer = Dense(512, activation='relu')(flatten_layer)
        output_layer = Dense(4,activation="linear")(dense1_layer)
        cnn = tf.keras.Model(inputs=input_layer, outputs=output_layer)


        # cnn = Sequential((84,84,state_len))
        # cnn.add(Input(shape=))
        # cnn.add(Conv2D(32, (8,8), strides=4, activation='relu'))
        # cnn.add(Conv2D(64, (4,4), strides=2, activation='relu'))
        # cnn.add(Conv2D(64, (3,3), strides=2, activation='relu'))
        # cnn.add(Flatten())
        # cnn.add(Dense(512, activation='relu'))
        # cnn.add(Dense(4,activation=None)) # no activation function (means the output is the sum of the input to the neuron)
        # cnn.compile(optimizer=Adam(learning_rate=adam_learning_rate), loss=Huber())
        return cnn


class Eps_Type(Enum):
    CONSTANT = 1
    ANNEALING_LINEAR = 2
    ANNEALING_STEPS = 3


class Training_History():
    def __init__(self, n_episodes) -> None:
        self.history = defaultdict(list)
        self.history['score'] = np.zeros(n_episodes)
        self.history['test_score'] = np.zeros(n_episodes)
        self.history['steps'] = np.zeros(n_episodes)
        self.history['time'] = np.zeros(n_episodes)
        self.history['episoln'] = np.zeros(n_episodes)
        self.history['avg_score_10'] = np.zeros(n_episodes)
        self.history['frames_hour'] = np.zeros(n_episodes)
        self.history['loss'] = np.zeros(n_episodes)
        self.history['std_score_10'] =  np.zeros(n_episodes)
    
    def update(self, ep, score, test_score, steps, eps, loss, verbose=0):
        self.history['score'][ep] = score
        self.history['test_score'][ep] = test_score
        self.history['steps'][ep] = steps
        self.history['episoln'][ep] = round(eps,3)
        self.history['time'][ep] = self.delta_time()
        self.history['avg_score_10'][ep] = self.calculate_moving_avg(10,ep)
        self.history['std_score_10'][ep] = self.calculate_moving_std(10,ep)
        self.history['frames_hour'][ep] = round(self.history['steps'][ep] / self.history['time'][ep] * 3600,0)
        self.history['loss'][ep] = loss
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
    
    def  calculate_moving_std(self,window,ep):
        return round(np.std(self.history['score'][(ep-window)*(ep>window):ep+1]),2)

    def print_summary(self, ep):
        print(f"Episode: {ep}, "
            f"Game score = {self.history['score'][ep]}, "
            f"Test score = {self.history['test_score'][ep]}, "
            f"Steps = {self.history['steps'][ep]}, "
            f"eps = {self.history['episoln'][ep]}, "
            f"10 games avg = {self.history['avg_score_10'][ep]}, "
            f"10 games std = {self.history['std_score_10'][ep]}, "
            f"Cumulative steps = {np.sum(self.history['steps'][0:ep+1])}, "
            f"Frames per hour = {np.sum(self.history['frames_hour'][ep])}, "
            f"loss = {self.history['loss'][ep]}")
    
    def save(self, file_name="history.pickle"):
        with open(file_name, 'wb') as file:
            pickle.dump(self.history, file)


class PreProcessing:
    
    def __init__(self, n_lives=5) -> None:
        self.lives = n_lives
        self.reward = 0
        self.clear_frame()
        self.clear_reward()
        self.clear_terminal()

    def clear_frame(self):
        self.frame = np.zeros((1,84,84,1))

    def clear_reward(self):
        self.reward = 0
    
    def clear_terminal(self):
        self.teminal = False 

    def preprocess_state(self, s):
        return s.reshape(1,84,84,1).astype(np.uint8)
    
    def preprocess_reward(self, r, info):
        if self.lives > info['lives']:
            r = -1
            self.lives = info['lives']
        self.reward +=r
    
    def preprocess_terminal(self, terminal, info):
        if self.lives > info['lives'] or terminal:
            self.teminal = True

    def overimporse_states(self, state, fading_rate):
        state = self.preprocess_state(state)
        self.frame = np.concatenate((self.frame ,state), axis=3) * fading_rate

    def get_overimposed_state(self):
        frame = np.max(self.frame, axis=3)
        frame = self.preprocess_state(frame)
        self.clear_frame()
        return frame

    def get_reward(self):
        reward = np.sign(self.reward) # cap all rewards between [-1, 1] 
        self.clear_reward()
        return reward
    
    def get_terminal(self):
        terminal = self.teminal # cap all rewards between [-1, 1] 
        self.clear_terminal()
        return terminal
        
    def reset(self,n_lives):
        self.lives = n_lives
        self.clear_frame()
        self.clear_reward()
        self.clear_terminal()



class Memory:
    def __init__(self,N=0, save=False):
        self.N = N
        self.buffer = [() for _ in range(N)]
        self.buffer[-1] = (np.zeros((1,84,84,1),dtype=np.uint8), 0, np.zeros((1,84,84,1),dtype=np.uint8), 0, 0) # set the last element in the memory as the default element.
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

    def get_experiences_in_batch(self, batch_size=16, batch_idxs=np.array([]) ,len_states=4):
        if batch_idxs.shape[0] == 0:
            batch_idxs =  np.random.randint(0, self.get_upper_bound(), batch_size)
        s_batch, obs_batch, a_batch, r_batch, t_batch = self.get_experiece_in_batch_by_idx(batch_idxs)
        terminal_states = np.zeros(batch_size)
        for k in range(1, len_states):
            # TODO  FINISH
            idxs = batch_idxs-k
            if self.is_full:
                idxs = ((idxs<0) * (self.get_upper_bound() -k)) + ((idxs>0) * idxs) # make sure I use the correct indexes.
            else:
                idxs = (idxs<0) * (-1) + (idxs>0) * idxs # if the buffer is not full

            s_batch_k, obs_batch_k, a_batch_k, r_batch_k, t_batch_k = self.get_experiece_in_batch_by_idx(idxs)
            terminal_states =  np.sign(terminal_states + np.abs(t_batch_k-1)) 
            s_batch_list = [s_batch_k[i].reshape(1,84,84,1) if terminal_states[i] == 0 else np.zeros((1,84,84,1),dtype=np.uint8) for i in range(batch_size)]
            s_batch_k = np.concatenate(s_batch_list, axis=0)
            
            s_batch = np.concatenate([s_batch, s_batch_k], axis = 3)

        return s_batch, obs_batch, a_batch, r_batch, t_batch

    def get_experiece_in_batch_by_idx(self, batch_idxs):
        s = []; a = [];  obs = []; r = []; t = []
        [[s.append(self.buffer[i][0]),
          a.append(self.buffer[i][1]),
          obs.append(self.buffer[i][2]),
          r.append(self.buffer[i][3]),
          t.append(self.buffer[i][4])] for i in batch_idxs] # removed unique. same sample can exist in the batch
        
        s_batch = np.concatenate(s,axis=0).astype(np.uint8)
        a_batch = np.array(a)
        obs_batch = np.concatenate(obs,axis=0).astype(np.uint8)
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
    env = AtariPreprocessing(env,frame_skip=1)
    #test_env(env)
    agent = DQN_Agent(eps_schema=Eps_Type.ANNEALING_LINEAR, memory_size=100000, save=True, verbose=1, folder="./GradTape_Approach/")
    agent.train(env, n_episodes=5000, frame_skip=4, batch_size=32)
    #agent.load_q_net("q_net_3000.h5")

    #agent.play(env, n_episodes=30)