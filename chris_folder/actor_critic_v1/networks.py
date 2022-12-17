import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Actor:
    """
    The Actor class is used to take actions in the environment based on policy

    :ivar network: references the Actor's neural network
    """

    def __init__(self) -> None:
        self.network = None

    def create_network(self, state_dims: np.array, num_actions: np.array, 
                       learning_rate: float) -> None:
        """ Initialise the neural network for the actor

        Parameters:
        state_dims (numpy array): shape of the observable environment
        num_actions (numpy array): Number of actions that the Actor must take

        Returns:
        None
        """
        
        model = Sequential()
        model.add(Dense(24, input_shape=state_dims, activation="relu"))
        model.add(Dense(12, input_shape=state_dims, activation="relu"))
        # For Bipedal Walker - output layer tanh gives 4 actions each in range
        # -1 to 1
        model.add(Dense(num_actions, activation="tanh"))
        optimiser = optimiser = Adam(learning_rate=learning_rate,
                                     metrics=["accuracy"])
        model.compile(loss="rmse")

        self.network = model

        return None
