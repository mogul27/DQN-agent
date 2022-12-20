import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from keras.initializers import GlorotNormal
import tensorflow_probability as tfp
import keras.backend as K

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
        model.compile(loss=self.actor_custom_loss, optimizer=optimiser)

        self.network = model

        return None

    def save_network_weights(self, game_type: str="box2d") -> None:
        """Save network weights to be loaded"""

        if game_type == "atari":
            self.network.save_weights("atari_actor_weights.h5")
        
        else:
            self.network.save_weights("actor_weights.h5")

        return None

    def load_network_weights(self, game_type: str="box2d") -> None:
        """Load previously saved weights for actor"""

        if game_type == "atari":
            self.network.load_weights("atari_actor_weights.h5")
        
        else:
            self.network.load_weights("actor_weights.h5")

        return None

    def predict(self, state: np.array, action_space: list) -> np.array:
        """Given an environment state, return an array of continuous actions
        with the mean and standard deviation of the distribution from which
        the actions are sampled.

        Parameters:
        state (np.array): environment state returned by env.step()
        action_space (list): List of actions available to the agent

        Returns:
        sampled_actions (np.array): Array of continuous actions for the agent to take
        """
        
        state = np.expand_dims(state, axis=0)
        probs_output = self.network.predict_on_batch(state)[0]
        print(probs_output)

        # actions
        action = np.random.choice(action_space, 1, p=probs_output)[0]

        return action

    def train(self, state: np.array, adv_function: np.array, action_taken:int,
              num_actions: int) -> np.array:
        """Adjust network parameters to perform Actor update step

        Parameters:
        state (np.array): environment state returned by env.step()
        adv_function (np.array): Advantage calculated using Critic valuation
        of action against state value
        action_take (int): action taken by the agent
        num_action (int): Number of available actions to the agent
        
        Returns:
        None
        """
        
        
        # Use delta and advantage as categorical crossentropy does the negative log part
        state = np.expand_dims(state, axis=0)
        softmax_probs = self.network.predict_on_batch(state)[0]

        # Create array and set action taken to 1 to one hot encode
        actual_action = np.zeros(num_actions)
        actual_action[action_taken] = 1
        gradient = actual_action - softmax_probs
        gradient_with_advantage =  adv_function * gradient + softmax_probs
        self.adv_function = adv_function
        # print("Encoded:", actual_action)
        # print("Probs:", softmax_probs)
        # print("gradient", gradient)
        # print("adv:", adv_function)
        # print("gaw", gradient_with_advantage)
        

        # state = np.expand_dims(state, axis=0)
        # probs_output = self.network.predict_on_batch(state)[0]
        # prob_dist = tfp.distributions.Categorical(probs=probs_output)
        # log_probs = prob_dist.log_prob(action)
        # actor_loss = -log_probs*adv_function*0.001  
        
        # self.network.train_on_batch(state, actor_loss)

        print(actual_action.reshape(1, 2))
        self.network.train_on_batch(state, gradient_with_advantage)

        return None

    def actor_custom_loss(self, y_true, y_pred):
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

        out = K.clip(y_pred, 1e-8, 1-1e-8) # set boundary
        log_lik = y_true * K.log(out) # policy gradient
        return K.sum(-log_lik * self.adv_function)
        
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


    def save_network_weights(self, game_type: str="box2d") -> None:
        """Save network weights to be loaded"""

        if game_type == "atari":
            self.network.save_weights("atari_critic_weights.h5")
        
        else:
            self.network.save_weights("critic_weights.h5")

        return None

    def load_network_weights(self, game_type: str="box2d") -> None:
        """Load previously saved weights for critic"""

        if game_type == "atari":
            self.network.load_weights("atari_critic_weights.h5")
        
        else:
            self.network.load_weights("critic_weights.h5")

        return None



class ConvActor(Actor):
    """Child class of Actor which uses te same methods but different
    functionality to be compatible with Atari game input
    """
    def __init__(self) -> None:
        super().__init__()

    def create_network(self, state_dims: tuple,
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

        initialiser = GlorotNormal()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=state_dims))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_actions, activation="softmax", kernel_initializer=initialiser))

        optimiser = Adam(learning_rate=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimiser)
        self.network = model

        return None

class ConvCritic(Critic):

    """ Initialise th Convlutional neural network for the Critic

    Parameters:
    state_dims (tuple): shape of the observable environment (default (24,))
    num_actions (int): Number of actions that the Actor must take (default 4)
    learning_rate (float): Learning rate for the Actor neural network (default 0.001)

    Returns:
    None
    """

    def __init__(self) -> None:
        super().__init__()
    
    def create_network(self, state_dims: tuple=(4,),
        learning_rate: float=0.001) -> None:

        initialiser = GlorotNormal()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=state_dims))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation="linear", kernel_initializer=initialiser))

        optimiser = Adam(learning_rate=learning_rate)
        model.compile(loss="mean_squared_error", optimizer=optimiser)

        self.network = model

        return None

    




