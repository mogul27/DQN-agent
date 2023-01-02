import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from keras.initializers import GlorotNormal, RandomUniform
import tensorflow_probability as tfp
import keras.backend as backend


class Actor:
    """
    The Actor class is used to take actions in the environment based on policy

    :ivar network: references the Actor's neural network
    :ivar entropy weight: Sets the weight applied to entropy in Actor loss function
    """

    def __init__(self, entropy_weight: float=0.5) -> None:
        self.network = None
        self.entropy_weight = entropy_weight

    def create_network(self, state_dims: tuple=(4,),
                       num_actions: int=2,
                       learning_rate: float=0.001) -> None:
        """ Initialise the neural network for the actor

        Parameters:
        state_dims (tuple): shape of the observable environment (default (4,))
        num_actions (int): Number of actions that the Actor must take (default 2)
        learning_rate (float): Learning rate for the Actor neural network (default 0.001)
        

        Returns:
        None
        """
        
        model = Sequential()
        model.add(Dense(24, input_shape=state_dims, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(num_actions, activation="softmax"))

        optimiser = Adam(learning_rate=learning_rate)

        # Use custom loss function defined in same file
        model.compile(loss=self.actor_custom_loss, optimizer=optimiser)

        self.network = model

        return None

    def save_network_weights(self, game_type: str="box2d", episode: int=0) -> None:
        """Save network weights to be loaded"""

        if game_type == "atari":
            self.network.save_weights("GoldenRunWeights/atari_actor_weights{}.h5".format(episode))
        
        else:
            self.network.save_weights("actor_weights.h5")

        return None

    def load_network_weights(self, game_type: str="box2d",
                             weights_path_override=None) -> None:
        """Load previously saved weights for actor
        
        Parameters
        game_type (str): Specifies what kind of environmnet the agent is operating in
        weights_path_override: Allows user to override default weight names to load weights
        stored under different names or in different locations
        """

        if weights_path_override:
            self.network.load_weights(weights_path_override)
            return None

        if game_type == "atari":
            self.network.load_weights("atari_actor_weights.h5")
        
        else:
            self.network.load_weights("actor_weights.h5")

        return None

    def predict(self, state: np.array, action_space: list) -> np.array:
        """Given an environment state and action space, return an action
        sampled from the actor probability distribution"

        Parameters:
        state (np.array): environment state returned by env.step()
        action_space (list): List of actions available to the agent

        Returns:
        action (np.array): Action for agent to take
        """
        
        # expand dimensions for CNN input
        state = np.expand_dims(state, axis=0)
        # Get the probability distribution returned by actor
        probs_output = self.network.predict_on_batch(state)[0]

        # Sample probability ditribution to get action
        action = np.random.choice(action_space, 1, p=probs_output)[0]

        return action

 
    def test_predict(self, state: np.array, action_space: list,
                     sample=True) -> np.array:
        """Given an environment state and action space, return an action
        sampled from the actor probability distribution
        Used when observing agent to give choice of sampling or maximising
        from actor distribution"

        Parameters:
        state (np.array): environment state returned by env.step()
        action_space (list): List of actions available to the agent
        sample (bool): Indicates whether to sample or take max action
        from action probabiliy distribution

        Returns:
        action (np.array): Action for agent to take

        """
        
        # Expand dimensions for CNN
        state = np.expand_dims(state, axis=0)
        # Get the probability distribution returned by actor
        probs_output = self.network.predict_on_batch(state)[0]

        if sample == True:
            # Sample from probability distribution
            action = np.random.choice(action_space, 1, p=probs_output)[0]
        else:
            # Take the action with the highest probability
            action = np.argmax(probs_output)

        return action

    def train(self, state: np.array, adv_function: np.array, action_taken:int,
              num_actions: int) -> np.array:
        """Adjust network parameters to perform Actor update step

        Parameters:
        state (np.array): environment state returned by env.step()
        adv_function (np.array): Advantage calculated using Critic valuation
        of action against state value
        action_taken (int): action taken by the agent
        num_action (int): Number of available actions to the agent
        
        Returns:
        None
        """
        
        
        # Expand state for input to CNN
        state = np.expand_dims(state, axis=0)
        # Retrieve the probability distribution of actions for a given state
        softmax_probs = self.network.predict_on_batch(state)[0]
        print(softmax_probs)

        # One hot encode the action taken from possible actions
        actual_action = np.zeros(num_actions)
        actual_action[action_taken] = 1

        # Process for obtaining deltheta follows an extended version of 
        # that in Wang (2021) - full reference in project report

        # Calculate deltheta (gradient)
        deltheta = actual_action - softmax_probs
        
        # calculate deltheta with Advantage (A(st, at))
        deltheta_advantage = adv_function * deltheta + softmax_probs
        
        # Train agent using custom loss function on state and deltheta_advantage
        self.network.train_on_batch(state, deltheta_advantage)

        return None

    def actor_custom_loss(self, y_true, y_pred):
        """Custom loss functon

        Parameters:
        y_true: Ground truth value for network
        y_pred: Predicted values from network
        
        Returns:
        actor_loss: Network loss calculated according to A2C loss function
        
        """

        clipped_vals = backend.clip(y_pred, 1e-7, 1-1e-7) # Clip before taking log of probs 

        # get log of probabilities for numerical stability and multiply by gradient with advantage
        log_likelihood =  backend.log(clipped_vals) * y_true
        # Include entropy bias to prevent agent always selecting same action 
        entropy = backend.sum(clipped_vals * backend.log(clipped_vals))
        # Calculate Actor Loss for A2C
        actor_loss = backend.sum(-log_likelihood) + (self.entropy_weight * entropy)

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
        
        # Expand dimensions for input to CNN
        state = np.expand_dims(state, axis=0)
        # Retrieve the value of the state from Critic network
        state_value = self.network.predict_on_batch(state)
        state = state_value[0]
        
        return state_value

    def train(self, state: np.array, td_target: np.array) -> np.array:
        """Adjust network parameters to perform Critic update step

        Parameters:
        state (np.array): environment state returned by env.step()
        td_target (np.array): TD learning update target
        
        Returns:
        None
        """

        # Expand dimensions for input to CNN
        state = np.expand_dims(state, axis=0)
        # Train critic network with state and td_target
        self.network.train_on_batch(state, td_target)


    def save_network_weights(self, game_type: str="box2d", episode: int=0) -> None:
        """Save network weights to be loaded"""

        if game_type == "atari":
            self.network.save_weights("GoldenRunWeights/atari_critic_weights{}.h5".format(episode))
        
        else:
            self.network.save_weights("critic_weights.h5")

        return None

    def load_network_weights(self, game_type: str="box2d",
                            weights_path_override=None) -> None:
        """Load previously saved weights for critic
        
        Parameters
        game_type (str): Specifies what kind of environmnet the agent is operating in
        weights_path_override: Allows user to override default weight names to load weights
        stored under different names or in different locations

        """
        
        if weights_path_override:
            self.network.load_weights(weights_path_override)
            return None

        if game_type == "atari":
            self.network.load_weights("GoldenRunWeights/atari_critic_weights.h5")
        
        else:
            self.network.load_weights("critic_weights.h5")

        return None



class ConvActor(Actor):
    """Inherits from Actor: uses te same methods but different
    functionality (Inclduing convolutional network) to be compatible
    with Atari game input

    :ivar network: references the Actor's neural network
    :ivar entropy weight: Sets the weight applied to entropy in Actor loss function

    """

    def __init__(self, entropy_weight=0.5) -> None:
        super().__init__()
        self.entropy_weight = entropy_weight

    def create_network(self, state_dims: tuple,
                       num_actions: int=2,
                       learning_rate: float=0.001) -> None:
        """ Initialise the neural network for the actor

        Parameters:
        state_dims (tuple): shape of the observable environment 
        num_actions (int): Number of actions that the Actor must take (default 2)
        learning_rate (float): Learning rate for the Actor neural network (default 0.001)

        Returns:
        None
        """

        # Use Random uniform with small window as Atari game sensitive to initial conditions
        initialiser = RandomUniform(minval=0, maxval=0.02)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=state_dims))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_actions, activation="softmax", kernel_initializer=initialiser))

        optimiser = Adam(learning_rate=learning_rate)

        # Use custom loss function defined in the same file
        model.compile(loss=self.actor_custom_loss, optimizer=optimiser)
        self.network = model

        return None

class ConvCritic(Critic):
    """Inherits from Critic class but uses a Convolutional network
    rather than feedforward

    """

    def __init__(self) -> None:
        super().__init__()
    
    def create_network(self, state_dims: tuple=(4,),
        learning_rate: float=0.001) -> None:
        """ Initialise th Convlutional neural network for the Critic

        Parameters:
        state_dims (tuple): shape of the observable environment (default (4,))
        learning_rate (float): Learning rate for the Actor neural network (default 0.001)

        Returns:
        None

        """

        # Use Random uniform with small window as Atari game sensitive to initial conditions
        initialiser = RandomUniform(minval=0, maxval=0.02)
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

    




