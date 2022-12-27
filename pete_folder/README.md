Contains the following implementations:
---------------------------------------

* [classic_dqn.py](classic_dqn.py) A DQN set up to run against gym's Cart-Pole
* [classic_dqn_torch.py](classic_dqn_torch.py) Same as classic_dqn.py, but using PyTorch rather than Keras
* [atari_dqn.py](atari_dqn.py) A DQN set up to run breakout from gym's Atari games
* [classic_asynch_q_learning.py](classic_asynch_q_learning.py) An asynchronous version of Q-Learning set up to run against Cart-Pole
* [atari_asynch_q_learning.py](atari_asynch_q_learning.py) An asynchronous version of Q-Learning set up to run against breakout

asynch q-learning
-----------------

The asynch code is implemented using PyTorch. It has it's own implementaion of
multiprocessing and allows network models to be easily shared. 

It does try and tidy up after itself, closing down all the processes. But, if you kill
the controller, then it's possible the child processes won't close down cleanly.

The _AsyncQLearningController_ does the following:
* Creates value and target networks
* Puts the networks in shared memory
* Creates queues to communicate with workers and stats collector
* Creates AsyncQLearnerWorker objects, passing them the value and target networks and suitable queues
* Starts each worker in it's own process
* Creates an AsyncStatsCollector object, passing it suitable queues
* Listens for episode complete messages from the workers:
  * For each end episode message, updates count of episodes and writes out the details
  * Every n episodes
    * Sends a message to the AsyncStatsCollector giving it the network weights
    * Saves the network weights
  * Every C episodes refresh the weights in the target network with those from the value network
  * After options['episodes'] number of episodes, stop all the workers and stats collector 
    
The _AsyncQLearnerWorker_ objects do the following:
* Run an episode. 
* Every n (default 5) steps, update the value network.
* At the end of an episode, send details to the AsyncQLearningController, then start a 
new episode.
* The also listen out for stop requests, and stop if they get one.

The _AsyncStatsCollector_ object does the following:
* Listen out for 'play' messages
  * When it gets one, it loads the supplied weights into a copy of the network and plays
  n episdoes, calculating the average reward and records the details.
* Listen out for 'stop' messages and stop if it gets one.  

**Options**: There are many including:
* _episodes_ = Number of episodes to run.
* _num_workers_ = Number of worker processes to start up.
* _worker_throttle_ = number of seconds worker sleeps after every value network update. 
This was purely to stop my machine melting when running multiple workers.
* _async_update_freq_ = Number of steps before the worker updates the value network.
* _play_avg_ = Number of games the stats collector runs each time to get an average.
* _stats_every_ = Number of episodes between gathering stats. Keep in mind that the 
stats collector has to run 'play_avg' episodes, so if stats_every / num_workers < play_avg
then it's likely that the stats collector will not be able to keep up.

 
