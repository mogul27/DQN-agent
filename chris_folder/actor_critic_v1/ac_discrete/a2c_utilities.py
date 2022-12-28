import numpy as np
import random
from collections import deque

class StateHistory:

    def __init__(self):
        """ Store tuples of head (current) state and its previous 3 frames and whether any of those
        frames are terminal to avoid including states running over differnt episodes.
        Contains a list of 4 tuples representing the current and 3 previous frames
        """
        
        self.data = []

    def empty_state(self):
        """Create an empty state """
        empty_state = np.zeros((84, 84))
        terminal = 0

        empty_history = [(empty_state, terminal) for i in range(4)]

        self.data = empty_history

    def get_states(self):
        """Retrieve only the states without the terminated flag to be passed into a cnn"""
        
        states = [state_terminal_pair[0] for state_terminal_pair in self.data]
        states = self.reshape_state(states)

        return states[0]

    
    def reshape_state(self, states):
        """Reformat a state with history to be fed into cnn"""

        states = np.array(states)
        reshaped_history = np.transpose(states)
        network_reshaped_history = reshaped_history.reshape(-1, 84, 84, 4)

        return network_reshaped_history
    
    def is_terminal(self):
        """Check terminal flag on most recent state in StateHistory"""
        return self.data[-1][1]

    def create_empty_frame(self):
        """Create a single empty frame"""

        empty_frame = np.zeros((84, 84))

        return empty_frame