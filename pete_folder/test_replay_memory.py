import unittest
import numpy as np
from dqn_utils import ReplayMemory, DataWithHistory


class MemoryReplayTest(unittest.TestCase):

    def test_add_and_get(self):
        replay_memory = ReplayMemory()
        for i in range(8):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)

        # We should just have the 8 added items
        self.assertEqual(8, replay_memory.size)

        # Get the 5th entry added.
        data_item = replay_memory.get_item(5)

        # Check we got state 5, and 3 history items of states 4, 3 and 2
        self.assertEqual(['state 5', 'state 4', 'state 3', 'state 2'], data_item.get_state())
        self.assertEqual(1, data_item.get_action())
        self.assertEqual(5, data_item.get_reward())
        self.assertEqual(['state 6', 'state 5', 'state 4', 'state 3'], data_item.get_next_state())
        self.assertEqual(False, data_item.is_terminated())

        # Get the last item added
        data_item = replay_memory.get_last_item()

        # Check we got state 7, and 3 history items of states 6, 5 and 4
        self.assertEqual(['state 7', 'state 6', 'state 5', 'state 4'], data_item.get_state())
        self.assertEqual(3, data_item.get_action())
        self.assertEqual(7, data_item.get_reward())
        self.assertEqual(['state 8', 'state 7', 'state 6', 'state 5'], data_item.get_next_state())
        self.assertEqual(False, data_item.is_terminated())

    def test_batch_retrieve(self):
        replay_memory = ReplayMemory()
        for i in range(8):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)

        batch = replay_memory.get_batch(batch_size=2)

        self.assertEqual(2, len(batch))
        for data_entry in batch:
            # history = 3, so should be 4 data items in each entry
            self.assertEqual(4, len(data_entry.get_state()))
            self.assertEqual(4, len(data_entry.get_next_state()))

    def test_with_no_history(self):
        replay_memory = ReplayMemory(history=0)
        for i in range(4):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)

        batch = replay_memory.get_batch(batch_size=3)

        self.assertEqual(3, len(batch))
        for data_entry in batch:
            # No history, so should be single data items
            self.assertEqual(1, len(data_entry.get_state()))
            self.assertEqual(1, len(data_entry.get_next_state()))

    def test_size_limit(self):
        replay_memory = ReplayMemory(max_len=4, history=0)
        # fill up the memory
        for i in range(4):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)
        self.assertEqual(4, replay_memory.size)

        data_item = replay_memory.get_item(0)
        self.assertEqual(('state 0', 0, 0, 'state 1', False), data_item.data[0])

        # Add another entry, should drop the first and add this to the end
        replay_memory.add(f"state new", 1, 0, f"state next", True)

        # first item should now be state 1
        data_item = replay_memory.get_item(0)
        self.assertEqual(('state 1', 1, 1, 'state 2', False), data_item.data[0])

        # last item should now be state new
        data_item = replay_memory.get_item(3)
        self.assertEqual(('state new', 1, 0, 'state next', True), data_item.data[0])


class DataWithHistoryTest(unittest.TestCase):

    def test_get_states(self):
        data = [(f"state {i}", i % 4, i, f"state {i+1}", False) for i in range(4)]
        data_with_history = DataWithHistory(data)

        state = data_with_history.get_state()

        self.assertEqual(['state 3', 'state 2', 'state 1', 'state 0'], state)


    def test_get_states_with_terminated(self):
        # mark 'state 1' as Terminated
        data = [(f"state {i}", i % 4, i, f"state {i+1}", i == 1) for i in range(4)]

        data_with_history = DataWithHistory(data)

        state = data_with_history.get_state()

        self.assertEqual('state 3', state[0])
        self.assertEqual('state 2', state[1])
        no_state = DataWithHistory.empty_state()
        self.assertTrue(np.array_equal(no_state, state[2]))
        self.assertTrue(np.array_equal(no_state, state[3]))



if __name__ == '__main__':
    unittest.main()
