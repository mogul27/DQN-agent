import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomUniform


class Actor:
    """
    The Actor class is used to take actions in the environment based on policy

    :ivar network: references the Actor's neural network
    """

    def __init__(self) -> None:
        self.network = None

    def create_network(self, state_dims: tuple=(24,),
                       num_actions: int=4,
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
        model.add(Dense(12, input_shape=state_dims, activation="relu"))
        # For Bipedal Walker - output layer tanh gives 4 actions each in range
        # -1 to 1, initialiser initialises output layer uniformly
        initialiser = RandomUniform(minval=-1.0, maxval=1.0)
        model.add(Dense(num_actions, activation="tanh",
                        kernel_initializer=initialiser))

        optimiser = Adam(learning_rate=learning_rate)
        model.compile(loss="rmse", optimizer = optimiser)

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

        Parameters:
        state (np.array): environment state returned by env.step()

        Returns:
        actions (np.array): Array of continuous actions for the agent to take
        """
        
        state = np.expand_dims(state, axis=0)
        actions = self.network.predict_on_batch(state)
        actions = actions[0] # Make actions 1-D array

        return actions

    def train(self, state: np.array, adv_function: np.array) -> np.array:
        """Adjust network parameters to perform Actor update step

        Parameters:
        state (np.array): environment state returned by env.step()
        adv_function (np.array): Advantage calculated using Critic valuation
        of action against state value
        
        Returns:
        None
        """

        self.network.train_on_batch(state, adv_function)

        return None

class Critic:
    """
    The Critic class evaluates the Actors choice of policy by mapping each
    state to it's corresponding Q-Value (Value of the state)

    :ivar network: references the Critic's neural network
    """

    def __init__(self) -> None:
        self.network = None

    def create_network(self, state_dims: tuple=(24,),
                       learning_rate: float=0.001) -> None:
        """ Initialise the neural network for the critic

        Parameters:
        state_dims (tuple): shape of the observable environment (default (24,))
        learning_rate (float): Learning rate for the Critic neural network (default 0.001)

        Returns:
        None
        """
        
        model = Sequential()
        model.add(Dense(24, input_shape=state_dims, activation="relu"))
        model.add(Dense(12, input_shape=state_dims, activation="relu"))
        model.add(Dense(1, activation="linear"))

        optimiser = Adam(learning_rate=learning_rate)
        model.compile(loss="rmse", optimizer = optimiser)

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

        self.network.train_on_batch(state, td_target)


    def save_network_weights(self) -> None:
       """Save network weights to be loaded"""
       self.network.save_weights("critic_weights.h5")

       return None

    def load_network_weights(self) -> None:
        """Load previously saved"""
        self.network.load_weights("critic_weights.h5")

        return None


