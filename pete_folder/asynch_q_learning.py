import multiprocessing as mp
import os
import time
import numpy as np
from queue import Empty

class QFunc:

    def __init__(self):
        self.msg = 'original'

    def display(self):
        print(f"process id: {os.getpid()} : says {self.msg}")

    def update(self, new_msg):
        self.msg = new_msg


class AsyncQLearnerWorker(mp.Process):

    def __init__(self, messages, network_gradient_queue):
        self.messages = messages
        self.network_gradient_queue = network_gradient_queue
        super().__init__()

    def run(self):
        print(f"I am a worker, my PID is {self.pid}")

        while True:
            # see if there are any messages
            if self.messages.poll():
                msg = self.messages.recv()
                print(f"worker {self.pid}: got message {msg}")

                self.network_gradient_queue.put(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


class AsyncQLearningController:

    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def train(self):
        print('parent process:', os.getppid())

        network_gradient_queue = mp.Queue(100)

        # set up the workers
        # TODO : should we make it spawn?
        workers = []
        for _ in range(self.num_workers):
            messages, child_conn = mp.Pipe()
            worker = AsyncQLearnerWorker(child_conn, network_gradient_queue)
            worker.daemon = True    # helps tidy up child processes if parent dies.
            worker.start()
            workers.append((worker, messages))

        data = ['one', 'two', 'three', np.array([[1], [2], [3]])]
        data_item = 0
        gradients_recieved = 0

        while data_item < len(data) or gradients_recieved < 4:
            if data_item < len(data):
                item = data[data_item]
                data_item += 1

                for worker, messages in workers:
                    # send message to each worker
                    print(f"controller sent item {item}")
                    messages.send(item)

            if gradients_recieved < 4:
                try:
                    network_gradient_deltas = network_gradient_queue.get(False)
                    print(f"controller got network gradients: {network_gradient_deltas}")
                    gradients_recieved += 1
                except Empty:
                    # Nothing to process, so just carry on
                    pass

        # close down the workers
        for worker, messages in workers:
            worker.terminate()
            worker.join(1)


def create_and_run_agent():
    agent = AsyncQLearningController()
    agent.train()


if __name__ == '__main__':
    create_and_run_agent()