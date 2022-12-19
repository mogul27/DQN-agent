import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras import Input, Model


class Actor:
    """
    The Actor class is used to take actions in the environment based on policy

    :ivar network: references the Actor's neural network
    """

    def __init__(self) -> None:
        self.network = None

    def create_network(self, state_dims: tuple=(4,),
                       num_actions: int=2,
                       learning_rate: float=0.001) -> None:
        """ Initialise the neural network for the actor

        Parameters:
        state_dims (tuple): shape of the observable environment (default (24,))
        num_actions (int): Number of actions that the Actor must take (default 4)
        learning_rate (float): Learning rate for the Actor neural network (default 0.001)

        Returns:
        None
        """
        
        model = Sequential()
        model.add(Dense(24, input_shape=state_dims, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(num_actions, activation="softmax"))

        optimiser = Adam(learning_rate=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimiser)

        self.network = model

        return None

    def save_network_weights(self) -> None:
       """Save network weights to be loaded"""
       self.network.save_weights("actor_weights.h5")

       return None

    def load_network_weights(self) -> None:
        """Load previously saved"""
        self.network.load_weights("actor_weights.h5")

        return None

    def predict(self, state: np.array) -> np.array:
        """Given an environment state, return an array of continuous actions
        with the mean and standard deviation of the distribution from which
        the actions are sampled.

        Parameters:
        state (np.array): environment state returned by env.step()

        Returns:
        sampled_actions (np.array): Array of continuous actions for the agent to take
        """
        
        state = np.expand_dims(state, axis=0)
        probs_output = self.network.predict_on_batch(state)[0]
        print(probs_output)

        action = np.random.choice([0, 1], 1, p=probs_output)[0]

        return action

    def train(self, state: np.array, adv_function: np.array, action:np.array) -> np.array:
        """Adjust network parameters to perform Actor update step

        Parameters:
        state (np.array): environment state returned by env.step()
        adv_function (np.array): Advantage calculated using Critic valuation
        of action against state value
        actions (np.array): actions taken by the agent
        
        Returns:
        None
        """

        state = np.expand_dims(state, axis=0)
        action_probs = self.network.predict_on_batch(state)[0]
        encoded_action = np.zeros(2)
        encoded_action[action] = 1
        gradient = encoded_action - action_probs
        gradient_with_advantage = .0001 * gradient * adv_function + action_probs

        # state = np.expand_dims(state, axis=0)
        # probs_output = self.network.predict_on_batch(state)[0]

        # log_probs = action_dist.log_prob(action)
        # print(log_probs)

        # actor_loss = -log_probs*adv_function  
        
        # print(actor_loss)
        self.network.train_on_batch(state, gradient_with_advantage)

        return None

    def actor_custom_loss(self, state, actor_loss):
        """Custom loss functon

        Parameters:
        y_pred (np.array): Predicted values from network
        
        Returns:
        ???
        
        """

        # state = np.expand_dims(state, axis=0)
        # mu, var = self.network.predict_on_batch(state)
        # # Make mu and var 1-D arrays
        # mu, var = mu[0], var[0]
        # std = np.sqrt(var) + 1e-5
        # sampled_actions = np.random.normal(mu, std)

        # actions = np.clip(actions, 1e-10, None)
        # log_probs = -np.log(actions)

        # actor_loss = log_probs*adv_function 
        
        return actor_loss

class Critic:
    """
    The Critic class evaluates the Actors choice of policy by mapping each
    state to it's corresponding Q-Value (Value of the state)

    :ivar network: references the Critic's neural network
    """

    def __init__(self) -> None:
        self.network = None

    def create_network(self, state_dims: tuple=(4,),
                       learning_rate: float=0.001) -> None:
        """ Initialise the neural network for the critic

        Parameters:
        state_dims (tuple): shape of the observable environment (default (24,))
        learning_rate (float): Learning rate for the Critic neural network (default 0.001)

        Returns:
        None
        """

        model = Sequential()
        model.add(Dense(240, input_shape=state_dims, activation="relu"))
        model.add(Dense(240, activation="relu"))
        model.add(Dense(1, activation="linear"))

        optimiser = Adam(learning_rate=learning_rate)
        model.compile(loss="mean_squared_error", optimizer = optimiser)

        self.network = model

        return None

    def predict(self, state: np.array) -> np.array:
        """Given an environment state, return a state value

        Parameters:
        state (np.array): environment state returned by env.step()

        Returns:
        state_value (np.array): Value for the given state
        """
        
        state = np.expand_dims(state, axis=0)
        state_value = self.network.predict_on_batch(state)
        state = state_value[0] # Make actions 1-D array
        
        return state_value

    def train(self, state: np.array, td_target: np.array) -> np.array:
        """Adjust network parameters to perform Critic update step

        Parameters:
        state (np.array): environment state returned by env.step()
        td_target (np.array): TD learning update target
        
        Returns:
        None
        """

        state = np.expand_dims(state, axis=0)
        self.network.train_on_batch(state, td_target)


    def save_network_weights(self) -> None:
       """Save network weights to be loaded"""
       self.network.save_weights("critic_weights.h5")

       return None

    def load_network_weights(self) -> None:
        """Load previously saved"""
        self.network.load_weights("critic_weights.h5")

        return None


