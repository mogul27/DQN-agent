from collections import deque
import random
import numpy as np


class ReplayMemory:
    """ Maintain a memory of states and rewards from previous experience.

    The replay_memory simply maintains a list of the (state, action, reward, next_state, terminated) combinations
    encountered while playing the game.
    It can then return a random selection from this list, and ass the memory is contiguous it can return the
    recent history too.
    """
    def __init__(self, max_len=1000, history=3):
        """
        :param max_len: The amount of items to hold in the memory
        :param history: Amount of history to include with each data entry returned : default 3
        """
        # TODO : Check if deque is best approach. It should be good for pop/append, but probably not so good
        #      : for the retrieve.
        self._data = deque()
        self.size = 0
        self.max_len = max_len
        self.history = history
        for _ in range(history):
            # Add some empty states to fill up the history to start.
            self.add(DataWithHistory.empty_state(), 0, 0, DataWithHistory, False)

    def add(self, state, action, reward, next_state, terminated):
        if self.size >= self.max_len:
            # don't grow bigger, just lose one off the front.
            self._data.popleft()
        else:
            self.size += 1
        self._data.append((state, action, reward, next_state, terminated))

    def get_item(self, index):
        """ Get a specific item from the replay memory along with the specified amount of history
        """
        if index < self.history:
            raise ValueError(f"Not enough history available for index {index}")

        return [self._data[i] for i in range(index - self.history,index + 1)]

    def get_batch(self, batch_size=1):
        """ get a batch of data entries. Returns a list of batch_size data items.

        Each data entry is a list of n contiguous state-action-reward experiences. n = history + 1

        :param batch_size: Number of data entries : default 1
        :return: list of lists of tuples of (state, action, reward, next_state)
        """
        if self.size < self.history + 1:
            raise ValueError(f"Not enough history available to be able to retrieve data. Size = {self.size}")

        batch = [self.get_item(random.randrange(self.history, self.size)) for i in range(batch_size)]

        return batch


class DataWithHistory:

    def __init__(self, data):
        """ data should be a list of (state, action, reward, next_state, terminated)

        This class provides a simple way to extract the state and next_state as lists of multiple items
        while also considering terminated. We don't want a history item that was terminated to be
        considered as a valid previous state.
        """
        # copy the supplied data.
        self.data = [item for item in data]
        self.data.reverse()

    @staticmethod
    def empty_state():
        no_state = np.zeros(128).reshape(8, 16)
        return no_state

    def get_states(self, state_field=0):
        states = None
        history_terminated = False
        for item in self.data:
            if states is None:
                states = [item[state_field]]
            else:
                history_terminated = history_terminated or item[-1]
                if history_terminated:
                    states.append(self.empty_state())
                else:
                    states.append(item[state_field])
        return states

    def get_next_states(self):
        # next_state is the 4th field, so pass in index of 3
        return self.get_states(3)



