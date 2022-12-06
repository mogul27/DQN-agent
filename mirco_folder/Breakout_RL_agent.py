

import gym
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from collections import defaultdict
from keras.models import Sequential     # ver 2.9.0
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input


# Create the environment
env = gym.make('ALE/Breakout-v5', render_mode="human")


# test the environment 
env.action_space.seed(42)
observation, info = env.reset(seed=42)
# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     if terminated or truncated:
#         observation, info = env.reset()




def DQN(env):
    n_episodes = 300
    n_actions = env.action_space.n # get the number of possible actions
    score_history =  np.zeros(n_episodes)
    #policy = defaultdict(lambda _ : np.ones * 1/ n_actions) # create an equiprobable policy for each unseen state.
    
    gamma = 0.98
    eps = 0.15

    n_replays = 5
    
    # intialize buffer B for experiece replay
    b_cap = 1000
    b_idx = 0
    b_full = False # flag to idicate whether the buffer has reached its capacity
    B =  [() for _ in range(b_cap)]
    
    # intialize Q_hat network
    q_net = keras.models.Sequential()
    q_net.add(Conv2D(32, (4,4), strides=(4,4), activation='relu', input_shape=(210,160,3)))
    q_net.add(Conv2D(64, (4,4), strides=(4,4), activation='relu'))
    q_net.add(Conv2D(64, (4,4), strides=(4,4), activation='relu'))
    q_net.add(Flatten())
    q_net.add(Dense(10, activation='relu'))
    q_net.add(Dense(4,activation=None)) # no activation function (means the output is the sum of the input to the neuron)
    q_net.compile(optimizer="adam", loss="mean_squared_error",metrics = "mean_squared_error")
    w = q_net.get_weights()

    # intiliaze Target net. I could probably just clone the net above.
    target_net = keras.models.Sequential()
    target_net.add(Conv2D(32, (4,4), strides=(4,4), activation='relu', input_shape=(210,160,3)))
    target_net.add(Conv2D(64, (4,4), strides=(4,4), activation='relu'))
    target_net.add(Conv2D(64, (4,4), strides=(4,4), activation='relu'))
    target_net.add(Flatten())
    target_net.add(Dense(10, activation='relu'))
    target_net.add(Dense(4,activation=None)) # no activation function (means the output is the sum of the input to the neuron)
    target_net.compile(optimizer="adam", loss="mean_squared_error",metrics = "mean_squared_error")
    target_net.set_weights(w) # make the target network the same as the q_network
    
    sync_steps = 30
    n_steps = 0

    for episode in range(n_episodes):
        state, info = env.reset()
        if episode % 5 == 0:
            model_name = "q_net_" + str(episode) + "_.h5"
            q_net.save(model_name) # save the q_network model every 20 episodes.

        terminal = False
        score = 0
        while not terminal:
            n_steps += 1

            if n_steps % sync_steps == 0: # syncronize the weigths of the target model with the q_net every [sync_steps]
                w = q_net.get_weights()
                target_net.set_weights(w)
            # TODO: must implement frame skipping. to speed up leaning and play more games.

            # take action and ovbserve next state and reward
            action = get_action(env, q_net,state.reshape(1,210,160,3), eps)
            #print(f'Action: {action}')
            observation, reward, terminal, truncated, info = env.step(action)
            score += reward

            #store observations in buffer.
            if b_idx == b_cap:
                if not b_full:
                    print("Buffer reached capacity") 
                b_idx = 0
                b_full = True
            B[b_idx] =  (state.reshape(1,210,160,3), action, observation.reshape(1,210,160,3), reward, terminal)
            b_idx +=1

            # update weights with the observation deriving from the current action
            update_weights(q_net, target_net, gamma, state.reshape(1,210,160,3), action, observation.reshape(1,210,160,3), reward, terminal) 

            # experience replay: performe temporaly uncorrelated elarning from k experiecese in the buffer
            for _ in range(n_replays):  # TODO: I could limit the nuber of replay when the buffer containes fewer number of experiences than the default number of replays
                rand_idx = np.random.randint(0, max(b_idx, b_full* len(B))) # this can be otpimized, (it's a waste calcualting the len every time)
                s, a, observed_s, r, term = B[rand_idx]
                update_weights(q_net, target_net, gamma, s, a, observed_s, r, term)

                
        print(f"Episode: {episode}, Total undiscouted reward = {score}")
        score_history[episode] = score
    return q_net, score_history


def update_weights(q_net, target_net,gamma, s, a, observed_s, r, terminal):
    # use the target net to compute the update target
    y = r +  (not terminal) * gamma * np.max(target_net.predict(observed_s, verbose=0))
 
    # use the q net to compute the estimated q values for all actions during the forward pass
    y_hat = q_net.predict(s, verbose=0)

    # overwirte the value for the taken action with the future expected return calculated by the target net
    y_hat[0][a] = y 

    # during the backward pass, update the weights of the net to reduce the rmse.
    # NOTE: only the action value for the current action  will contribute to the loss.
    # therefore the weights will be adjusted accordingly
    q_net.fit(s,y_hat, verbose=0) # TODO Instead of fitting the results of each replay independently, I could first simaulte all the rreply then fit the network cosidering all datapoints at once. It should be more effient



            
def get_action(env, q_net, state, eps,): # This can be made more efficient. draw p, if p < eps env.action_state.select() else predict and get the action with the highest value
    p = np.random.random()
    if p <= eps:
        action = env.action_space.sample()
    else:
        q_vals= q_net.predict(state, verbose=0)
        action = np.argmax(q_vals)
    return action


    # # estimatet the action value function for all actions in state s
    # q_vals= q_net.predict(state, verbose=0)
    # n_action = q_vals.shape[1]
    # # get the action with the highest expected return
    # A = np.argmax(q_vals)
    # p = np.random.random()
    # #  apply e-greedy soft policy 
    # action_probs = np.ones(n_action) * eps/n_action
    # action_probs[A] += (1-eps) 
    # # slect action a with probability p
    # cumprob = np.cumsum(action_probs)
    # action = np.where(cumprob >= p)[0][0]
    # return action


net, score_history = DQN(env)
net.save("q_net.h5")
