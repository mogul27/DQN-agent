from collections import deque
import random
import numpy as np
import time
import pickle
import os


class ReplayMemory:
    """ Maintain a memory of states and rewards from previous experience.

    The replay_memory simply maintains a list of the (state, action, reward, next_state, terminated) combinations
    encountered while playing the game.
    It can then return a random selection from this list, and ass the memory is contiguous it can return the
    recent history too.
    """
    def __init__(self, max_len=10000, history=3, file_name=None):
        """
        :param max_len: The amount of items to hold in the memory
        :param history: Amount of history to include with each data entry returned : default 3
        :param file_name: name of file to load/save replay memory from/to. If the file exists, then it is loaded
        during init and overrides max_len, history, etc. If no file exists, a new memory is created with the
        supplied values.
        """
        # TODO : Check if deque is best approach. It should be good for pop/append, but probably not so good
        #      : for the retrieve.
        self.file_name = file_name
        if file_name is None or not os.path.isfile(file_name):
            self._data = deque()
            self.size = 0
            self.max_len = max_len
            self.history = history
            for _ in range(history):
                # Add some empty states to fill up the history to start.
                self._data.append((DataWithHistory.empty_state(), 0, 0, DataWithHistory.empty_state(), False))
        else:
            self.load()

    def add(self, state, action, reward, next_state, terminated):
        if self.size >= self.max_len:
            # don't grow bigger, just lose one off the front.
            self._data.popleft()
        else:
            self.size += 1
        self._data.append((state, action, reward, next_state, terminated))

    def get_item(self, index):
        """ Get a specific item from the replay memory along with the specified amount of history

        :return: A DataWithHistory object
        """

        return DataWithHistory([self._data[i] for i in range(index, index + self.history + 1)])

    def get_last_item(self):
        """ Get the last item added
        """
        return self.get_item(self.size - 1)

    def get_batch(self, batch_size=1):
        """ get a batch of data entries. Returns a list of batch_size data items.

        Each data entry is a list of n contiguous state-action-reward experiences. n = history + 1

        :param batch_size: Number of data entries : default 1
        :return: list of lists of tuples of (state, action, reward, next_state)
        """

        batch = [self.get_item(random.randrange(0, self.size)) for i in range(batch_size)]

        return batch

    def save(self):
        if self.file_name is not None:
            with open(self.file_name, 'wb') as file:
                pickle.dump(self.size, file)
                pickle.dump(self.max_len, file)
                pickle.dump(self.history, file)
                pickle.dump(self._data, file)

    def load(self):
        if self.file_name is not None and os.path.isfile(self.file_name):
            with open(self.file_name, 'rb') as file:
                self.size = pickle.load(file)
                self.max_len = pickle.load(file)
                self.history = pickle.load(file)
                self._data = pickle.load(file)


class DataWithHistory:
    # TODO : add some constants for the field names.

    def __init__(self, data):
        """ data should be a list of (state, action, reward, next_state, terminated)

        This class provides a simple way to extract the state and next_state as lists of multiple items
        while also considering terminated. We don't want a history item that was terminated to be
        considered as a valid previous state.
        """
        # copy the supplied data.
        self.data = [item for item in data]

    @staticmethod
    def empty_state():
        return np.zeros((84, 84))

    def _states(self, state_field=0):
        states = [item[state_field] for item in self.data]
        # If any of the history is terminated, then clear the states for them
        history_terminated = False
        for i in range(len(self.data)-2, -1, -1):
            history_terminated = history_terminated or self.data[i][-1]
            if history_terminated:
                states[i] = self.empty_state()
        return states

    def get_state(self):
        return self._states(0)

    def get_action(self):
        return self.data[-1][1]

    def get_reward(self):
        return self.data[-1][2]

    def get_next_state(self):
        # next_state is the 4th field, so pass in index of 3
        return self._states(3)

    def is_terminated(self):
        return self.data[-1][4]


class Timer:
    """ Simple timer that accumulates time for an event.
     timer.start('event_name')
     timer.stop('event_name')

     timer.event_timer dict has the time of the last start/stop
     timer.cumulative_times dict has the accumulation of all start/stop for same event
     """

    def __init__(self):
        self.event_start = {}
        self.cumulative_times = {}
        self.event_times = {}

    def start(self, name):
        self.event_start[name] = time.time()

    def stop(self, name):
        if name in self.event_start:
            elapsed = time.time() - self.event_start[name]
            if name not in self.event_times:
                self.event_times[name] = 0.0
            self.event_times[name] = elapsed
            if name not in self.cumulative_times:
                self.cumulative_times[name] = 0.0
            self.cumulative_times[name] += elapsed

    def display_cumulative(self):
        for name, elapsed in self.cumulative_times.items():
            print(f"{name} : {elapsed:0.1f} seconds")

    def display_last(self):
        for name, elapsed in self.event_times.items():
            print(f"{name} : {elapsed:0.1f} seconds")


class Options:

    def __init__(self, values=None):
        """ Copies names and values from supplied Dict / Options object.

        :param values: dict with names and values, or an Option object.
        """
        if values is None:
            self.values = {}
        else:
            if isinstance(values, Options):
                values = values.values
            self.values = {name: value for name, value in values.items()}

    def default(self, name, value):
        """ Sets the value if the name not already in options, otherwise leave the value for name as is.
        """
        if name not in self.values:
            self.set(name, value)

    def get(self, name):
        """ Get value for name, return None if the name not in options.
        """
        if name in self.values:
            return self.values[name]
        return None

    def set(self, name, value):
        """ Set the value for the option name
        """
        self.values[name] = value


