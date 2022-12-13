import copy
import numpy as np
from dqn_agent import DQNAgent
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from dqn_utilities import ExperienceBuffer

def fill_replay_buffer(experience_buffer_size: int, possible_actions: list
                    ,random_history: ExperienceBuffer, agent: DQNAgent
                    ,wrapped_env: AtariPreprocessing):

    # Set terminal to False initially for looping
    terminal=False

    # Fill replay buffer sith initial random wandering
    for _ in range(int(experience_buffer_size/2)):
        action = np.random.choice(possible_actions)
        next_state, reward, terminal, _, _ = wrapped_env.step(action)
        
        # Create a copy of the state history object so it is not mutated in memory
        # then put the next state into the original object
        random_buffer_history = copy.deepcopy(random_history)
        random_history.data.pop(0)
        random_history.data.append((next_state, terminal))
        # Make a copy of next_state history object to store in memory to avoid mutating it
        next_random_buffer_history = copy.deepcopy(random_history)

        # Add experience to the buffer
        agent.experience_buffer.add(random_buffer_history, action, reward, next_random_buffer_history)

        if terminal:
            _, _ = wrapped_env.reset()