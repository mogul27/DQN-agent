import numpy as np
import random
from collections import deque

class ExperienceBuffer:
    """ Maintain a memory of states and rewards from previous experience.

    Stores (history, action, reward, next_history) combinations encountered while playing the game up
    to a limit N
    Return random selection from the list
    """

    def __init__(self, max_buffer_len: int):
        self.data = deque()
        self.size = 0
        self.max_len = max_buffer_len

    def add(self, history, action, reward, next_history):
        if self.size >= self.max_len:
            # don't grow bigger, just lose one off the front.
            self.data.popleft()
        else:
            self.size += 1
        self.data.append((history, action, reward, next_history))

    def get_random_data(self, batch_size=1) -> list:
        """get a random batch of data"""

        batch = random.choices(self.data, k=batch_size)

        return batch

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

        return states

    
    def reshape_state(self, states):
        """Reformat a state with history to be fed into cnn"""

        states = np.array(states)
        stacked_history = np.stack(states, axis=0)
        reshaped_history = np.transpose(stacked_history)
        network_reshaped_history = reshaped_history.reshape(1, 84, 84, 4)

        return network_reshaped_history
    
    def is_terminal(self):
        """Check terminal flag on most recent state in StateHistory"""
        return self.data[-1][1]

    def create_empty_frame(self):
        """Create a single empty frame"""

        empty_frame = np.zeros((84, 84))

        return empty_frame
    
    def stop_terminal_continuity(self):
        """If a state within a history is terminal, replace all states from that frame forward
        to prevent continuity issues across a single history"""

        # Create ordered list of terminal flags and find index of first one (default behaviour of .index)
        terminal_list = [state_terminal_pair[1] for state_terminal_pair in self.data]
        
        # create empty frame to replace terminal frames
        empty_frame = self.create_empty_frame()

        if 1 in terminal_list:
            terminal_index = terminal_list.index(1)
            # Replace terminal index onwards with empty frames
            for i in range(4-terminal_index, 5):
                self.data[-i] = (empty_frame, 0)