import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
from keras.models import Sequential     # ver 2.9.0
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import time
import pickle
from enum import Enum
from collections import defaultdict

""" Deprecated implementation. Will be deleted soon """

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


def DQN(env, q_net = [], target_net = [], n_steps=0, B=[]):
    SAVE_MODELS = True
    n_episodes = 2000
    score_history =  np.zeros(n_episodes)
    
    # hyperparameters
    gamma = 0.98
    eps_start = 1
    eps_end = 0.1
    steps_thr = 10e3 # threshold number of frames at which the exploration rate eps start decreasing  
    n_replays = 16

    learning_start = False
    # BUFFER
    if len(B) == 0:
        # intialize buffer B for experiece replay
        b_full = False # flag to idicate whether the buffer has reached its capacity
        b_cap = 200000
        B =  [() for _ in range(b_cap)]
    else: # use the buffer from a previosu run
        b_cap =  len(B)
        b_full = () in B
    b_idx = 0

    # intialize Q network
    if not isinstance(q_net,Sequential):
        q_net = target_net = intialize_CNN()
    w = q_net.get_weights()

    # intiliaze Target net.
    if not isinstance(target_net,Sequential):
        target_net = intialize_CNN()
    target_net.set_weights(w) # make the target network the same as the q_network
    
    sync_steps = 300

    time_start  = time.time()
    for episode in range(n_episodes):
        state, info = env.reset()
        state = state.reshape(1,84,84,1)
        if n_steps > steps_thr and episode % 20 == 0 and SAVE_MODELS:
            q_model_name = "q_net_" + str(episode) + ".h5"
            target_model_name = "taget_net_" + str(episode) + ".h5"
            q_net.save(q_model_name) # save the q_network model every 20 episodes.
            target_net.save(target_model_name) # save the target_network model every 20 episodes.

        terminal = False
        score = 0
        lives = info['lives']
        ep_steps = 0
        n_noop = 0
        while not terminal:
            n_steps += 1 # total steps (total game frames)
            ep_steps +=1 # step in this episode 
            # Syncronize the weigths of the target model with the q_net every [sync_steps]
            if n_steps % sync_steps == 0: 
                w = q_net.get_weights()
                target_net.set_weights(w)
            # TODO: must implement frame skipping. to speed up leaning and play more games.

            # Select an action deriving from q(s,a)
            eps = calculate_eps(eps_start, eps_end, n_steps, steps_thr) # annealing exploration rate
            action = get_action(env, q_net, state, eps, info['episode_frame_number'], n_noop)
            if action == 0:
                n_noop +=1
            else:
                n_noop = 0
            
            # Take action and ovbserve next state and reward
            observation, reward, terminal, truncated, info = env.step(action)
            observation = observation.reshape(1,84,84,1)
            
            # Every time a life is lost give a reward of -1.
            if info['lives'] < lives: 
                reward = -1
            lives = info['lives']
            score += reward

            # Store observations in buffer.
            if b_idx == b_cap:
                if not b_full:
                    print("Buffer reached capacity") 
                b_idx = 0
                b_full = True
            B[b_idx] =  (state, action, observation, reward, int(not terminal))
            b_idx +=1

            # Experience replay: perform  uncorrelated larning from k experiences in the buffer
            if n_steps > steps_thr: # start learnig only after a certain number of frames
                if not learning_start:
                    print('LEARNING STARTS NOW')
                    learning_start = True
                batch_idxs =  np.random.randint(0, max(b_idx, b_full* len(B)), n_replays)
                s = []; a = [];  obs = []; r = []; t = []
                [[s.append(B[i][0]), a.append(B[i][1]), obs.append(B[i][2]), r.append(B[i][3]), t.append(B[i][4])] for i in np.unique(batch_idxs)]
                s_batch = np.concatenate(s,axis=0)
                action_batch = np.array(a)
                obeservation_batch = np.concatenate(obs,axis=0)
                reward_batch = np.array(r)
                terminal_batch = np.array(t)
                update_weights(q_net, target_net, gamma, s_batch, action_batch, obeservation_batch, reward_batch, terminal_batch)
            state = observation  
        
        score_history[episode] = score        
        step_hour =   round(n_steps/((time.time()-time_start)/3600),0)      
        print(f"Episode: {episode}, Total undiscouted reward = {score}, Steps = {ep_steps}, Steps/hour = {step_hour}, eps = {round(eps,3)}, 10 games avg score = {round(np.mean(score_history[(episode-10)*(episode>10):episode+1]),2)}. Cumulative steps = {n_steps}")
       
    print(f'Total number of frames: {n_steps}')
    return q_net, target_net, score_history, n_steps, B


def intialize_CNN(learninng_rate = 0.001):
    cnn = Sequential()
    cnn.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(84,84,1)))
    cnn.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
    cnn.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dense(4,activation=None)) # no activation function (means the output is the sum of the input to the neuron)
    cnn.compile(optimizer="adam", loss="mean_squared_error",metrics = "mean_squared_error")
    return cnn



def calculate_eps(eps_start, eps_end, steps, steps_thr):
    steps_sat = 90000
    if steps < steps_thr: # for 50k frames selectes random actions.
        eps = eps_start
    elif steps > steps_sat:
        eps = eps_end
    else:
        eps = eps_start - (eps_start - eps_end) /steps_sat * steps
    return eps



def update_weights(q_net, target_net,gamma, s, a, observed_s, r, not_terminal):
    # use the target net to compute the update target
    y = r +  not_terminal * gamma * np.max(target_net.predict_on_batch(observed_s), axis=1)
 
    # use the q net to compute the estimated q values for all actions during the forward pass
    y_hat = q_net.predict_on_batch(s)

    # overwirte the value for the taken action with the future expected return calculated by the target net
    y_hat[range(y_hat.shape[0]),a] = y
    # during the backward pass, update the weights of the net to reduce the rmse.
    # NOTE: only the action value for the current action  will contribute to the loss.
    # therefore the weights will be adjusted accordingly
    q_net.train_on_batch(s,y_hat) # TODO Instead of fitting the results of each replay independently, I could first simaulte all the rreply then fit the network cosidering all datapoints at once. It should be more effient



            
def get_action(env, q_net, state, eps, ep_frame, n_noop): # This can be made more efficient. draw p, if p < eps env.action_state.select() else predict and get the action with the highest value
    n_noop_max = 30
    if ep_frame == n_noop and n_noop > n_noop_max:
        action =  np.random.randint(1,4)
        print(f'{n_noop_max} consecutive NOOP actions at the start of the episode: Action overwritten')
        return
    p = np.random.random()
    if p <= eps:
        action = env.action_space.sample()
    else:
        q_vals= q_net.predict_on_batch(state)
        action = np.argmax(q_vals)
    return action


def test_agent(env,agent,n_games):
    env.render_mode = 'human'
    scores  = np.zeros(n_games)
    for episode in range(n_games):
        state, info = env.reset()
        terminal = False
        game_score = 0
        while not terminal:
            action  = get_action(env,agent,state,0)
            observation, reward, terminal, truncated, info = env.step(action)
            state = observation
            game_score += reward
        scores[episode] = game_score
    return scores



# Create the environment
env = gym.make('ALE/Breakout-v5', render_mode=None, frameskip=1)
env = AtariPreprocessing(env,frame_skip=4)
#test_env(env)

q_net, target_net, score_history, n_frames, Buffer = DQN(env)

with open('buffer.pickle', 'wb') as file:
    pickle.dump(Buffer, file)
 # save he q and target net
q_net.save("q_net.h5")
target_net.save("target_net.h5")


