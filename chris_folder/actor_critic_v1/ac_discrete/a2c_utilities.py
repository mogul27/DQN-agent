import numpy as np
import random
from collections import deque

class StateHistory:

    def __init__(self):
        """ Stores lists of head (current) state and its previous 3 frames 
        
        Parameters:
        :ivar data: List containing current and previous frames
        """
        
        self.data = []

    def empty_state(self):
        """Fill the data list with empty state"""
        empty_state = np.zeros((84, 84))

        empty_history = [empty_state for i in range(4)]

        self.data = empty_history

    def get_states(self):
        """Retrieve states from StateHistory and prepare to feed into CNN"""
        
        states = self.reshape_state(self.data)

        return states[0]

    
    def reshape_state(self, states):
        """Reformat a state with history to be fed into cnn"""

        states = np.array(states)
        reshaped_history = np.transpose(states)
        network_reshaped_history = reshaped_history.reshape(-1, 84, 84, 4)

        return network_reshaped_history

    def create_empty_frame(self):
        """Create a single empty frame"""

        empty_frame = np.zeros((84, 84))

        return empty_frame