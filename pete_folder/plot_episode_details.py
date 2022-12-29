import matplotlib.pyplot as plt

import csv
from pathlib import Path
from dqn_utils import Options


def plot_rewards(x, x_label, rewards, title, file_name):
    # Same again, but by steps
    plots = []
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('reward')
    ax.grid(True)
    p, = ax.plot(x, rewards, marker='.', linestyle='', label='rewards')
    plots.append(p)

    avg_rwd = []
    last_few = []
    for reward in rewards:
        if len(last_few) >= 40:
            last_few.pop(0)
        last_few.append(reward)
        avg_rwd.append(sum(last_few)/len(last_few))
    p, = ax.plot(x, avg_rwd, label='moving average')
    plots.append(p)

    ax.legend(handles=plots,loc='upper left')

    fig.tight_layout()

    plt.savefig(file_name)


def read_data_and_plot_rewards(options):

    # location for input files.
    data_dir = Path(options.get('data_dir'))
    if not data_dir.exists():
        raise ValueError(f"Could not find data_dir {options.get('data_dir')}")

    # location for output files.
    out_dir = Path(options.get('out_dir'))
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Gather data from episode detail files.
    p = data_dir.glob('episode-detail*.csv')
    csv_files = [x for x in p if x.is_file()]
    print(csv_files)

    if len(csv_files) == 0:
        print(f"No files found in {data_dir} with pattern 'episode-detail*.csv'.")
        return

    csv_data = []

    for f in csv_files:
        with open(f, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                csv_data.append((int(row['episode']), int(row['steps']), float(row['reward'])))

    csv_data = sorted(csv_data)

    episode_detail = {}
    total_steps = 0
    for episode, steps, reward in csv_data:
        total_steps += steps
        episode_detail[episode] = (total_steps, reward, steps)

    # Now get the rewards from the play
    p = data_dir.glob('info*.csv')
    csv_files = [x for x in p if x.is_file()]
    print(csv_files)

    if len(csv_files) == 0:
        print(f"No files found in {data_dir} with pattern 'info*.csv'.")
        return

    csv_data = []

    for f in csv_files:
        with open(f, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                csv_data.append((int(row['episode']), float(row['reward'])))

    csv_data = sorted(csv_data)

    episodes = []
    rewards = []
    cumulative_steps = []

    for episode, reward in csv_data:
        episodes.append(episode)
        rewards.append(reward)
        if episode in episode_detail:
            cumulative_steps.append(episode_detail[episode][0])
        else:
            cumulative_steps.append(None)

    plot_filename = Path(options.get('plot_filename', 'async_rewards.png'))
    file_name = plot_filename.stem + ' by episode' + plot_filename.suffix
    title = options.get('plot_title', 'Asynch Q-Learning')
    plot_rewards(episodes, 'episodes', rewards, title, out_dir / file_name)

    # Same again, but by steps

    plot_filename = Path(options.get('plot_filename', 'async_rewards.png'))
    file_name = plot_filename.stem + ' by steps' + plot_filename.suffix
    title = options.get('plot_title', 'Asynch Q-Learning')
    cumulative_steps_per_100k = [cs/1000000 for cs in cumulative_steps]
    plot_rewards(cumulative_steps_per_100k, 'steps (millions)', rewards, title, out_dir / file_name)


    plt.close('all')    # matplotlib holds onto all the figures if we don't close them.


if __name__ == '__main__':

    read_data_and_plot_rewards(Options({
        'out_dir': "data/async_q_learning/",
        'data_dir': "data/async_q_learning/w16 lr0_0004/",
        'plot_filename': f'async_q_learning_rewards_breakout 3.png',
        'plot_title': f"Async Q-Learning Breakout 16 workers"
    }))
