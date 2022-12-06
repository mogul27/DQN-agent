import unittest
import numpy as np
from dqn_utils import ReplayMemory, DataWithHistory


class MemoryReplayTest(unittest.TestCase):

    def test_add_and_get(self):
        replay_memory = ReplayMemory()
        for i in range(8):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)

        # Should have 3 empty history, plus the 8 added items
        self.assertEqual(11, replay_memory.size)

        # Get the 5th entry added (need to add 3 when selecting to account for the empty states that are
        # added at the start as the history.)
        data_entry = replay_memory.get_item(8)
        self.assertEqual(4, len(data_entry))

        # Check we got state 5, and 3 history items of states 4, 3 and 2
        self.assertEqual(('state 5', 1, 5, 'state 6', False), data_entry[-1])

        self.assertEqual(('state 4', 0, 4, 'state 5', False), data_entry[-2])
        self.assertEqual(('state 3', 3, 3, 'state 4', False), data_entry[-3])
        self.assertEqual(('state 2', 2, 2, 'state 3', False), data_entry[-4])

    def test_batch_retrieve(self):
        replay_memory = ReplayMemory()
        for i in range(8):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)

        batch = replay_memory.get_batch(batch_size=2)

        self.assertEqual(2, len(batch))
        for data_entry in batch:
            # history = 3, so should be 4 data items in each entry
            self.assertEqual(4, len(data_entry))

    def test_with_no_history(self):
        replay_memory = ReplayMemory(history=0)
        for i in range(4):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)

        batch = replay_memory.get_batch(batch_size=3)

        self.assertEqual(3, len(batch))
        for data_entry in batch:
            # No history, so should be single data items
            self.assertEqual(1, len(data_entry))

    def test_size_limit(self):
        replay_memory = ReplayMemory(max_len=4, history=0)
        # fill up the memory
        for i in range(4):
            replay_memory.add(f"state {i}", i % 4, i, f"state {i+1}", False)
        self.assertEqual(4, replay_memory.size)

        data_item = replay_memory.get_item(0)
        self.assertEqual(('state 0', 0, 0, 'state 1', False), data_item[0])

        # Add another entry, should drop the first and add this to the end
        replay_memory.add(f"state new", 1, 0, f"state next", True)

        # first item should now be state 1
        data_item = replay_memory.get_item(0)
        self.assertEqual(('state 1', 1, 1, 'state 2', False), data_item[0])

        # last item should now be state new
        data_item = replay_memory.get_item(3)
        self.assertEqual(('state new', 1, 0, 'state next', True), data_item[0])


class DataWithHistoryTest(unittest.TestCase):

    def test_get_states(self):
        data = [(f"state {i}", i % 4, i, f"state {i+1}", False) for i in range(4)]
        data_with_history = DataWithHistory(data)

        states = data_with_history.get_states()

        self.assertEqual(['state 3', 'state 2', 'state 1', 'state 0'], states)


    def test_get_states_with_terminated(self):
        # mark 'state 1' as Terminated
        data = [(f"state {i}", i % 4, i, f"state {i+1}", i == 1) for i in range(4)]

        data_with_history = DataWithHistory(data)

        states = data_with_history.get_states()

        self.assertEqual('state 3', states[0])
        self.assertEqual('state 2', states[1])
        no_state = DataWithHistory.empty_state()
        self.assertTrue(np.array_equal(no_state, states[2]))
        self.assertTrue(np.array_equal(no_state, states[3]))



if __name__ == '__main__':
    unittest.main()
