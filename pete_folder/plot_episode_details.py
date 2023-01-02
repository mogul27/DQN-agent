import matplotlib.pyplot as plt

import csv
from pathlib import Path
from dqn_utils import Options


def plot_rewards_by_episode(all_data, title, file_name, include_scatter):
    # Same again, but by steps
    plots = []
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for name, data in all_data.items():
        x = data['episodes']
        rewards = data['rewards']

        if include_scatter:
            # Scatter plot for rewards
            ax.set_title(title)
            ax.set_xlabel('episodes')
            ax.set_ylabel('reward')
            ax.grid(True)
            p, = ax.plot(x, rewards, marker='.', linestyle='', label=f'{name} rewards')
            plots.append(p)

        # Moving average - start from 0?
        avg_rwd = []
        last_few = [0] * 10
        for reward in rewards:
            if len(last_few) >= 40:
                last_few.pop(0)
            last_few.append(reward)
            avg_rwd.append(sum(last_few)/len(last_few))
        p, = ax.plot(x, avg_rwd, label=f'{name}')
        plots.append(p)

        ax.legend(handles=plots, loc='lower right')

        fig.tight_layout()

        plt.savefig(file_name)


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
    last_few = [0, 0, 0, 0, 0]
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


def plot_all_episodes(episode_data, episode_detail, title, out_dir, plot_filename):
    plots = []
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.grid(True)

    epsilons = {epsilon for episode, steps, reward, pid, epsilon in episode_data}
    epsilons = list(epsilons)
    epsilons.sort()
    for eps in epsilons:
        worker_data = [(episode, reward) for episode, steps, reward, pid, epsilon in episode_data if eps == epsilon]
        episodes = [data[0] for data in worker_data]
        rewards = [data[1] for data in worker_data]
        # p, = ax.plot(episodes, rewards, marker='.', linestyle='', label='rewards')
        # plots.append(p)
        avg_rwd = []
        last_few = []
        for reward in rewards:
            if len(last_few) >= 40:
                last_few.pop(0)
            last_few.append(reward)
            avg_rwd.append(sum(last_few)/len(last_few))
        p, = ax.plot(episodes, avg_rwd, label=f'Epsilon {eps:0.2f}')
        plots.append(p)

    ax.legend(handles=plots, loc='lower right')

    file_name = Path(plot_filename.stem + ' all episodes aa' + plot_filename.suffix)
    plt.savefig(out_dir / file_name)

    plots = []
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel('steps (millions)')
    ax.set_ylabel('reward')
    ax.grid(True)

    epsilons = {epsilon for episode, steps, reward, pid, epsilon in episode_data}
    epsilons = list(epsilons)
    epsilons.sort()
    for eps in epsilons:
        worker_data = [(episode, reward) for episode, steps, reward, pid, epsilon in episode_data if eps == epsilon]
        steps = [episode_detail[data[0]][0] / 1000000 for data in worker_data]
        rewards = [data[1] for data in worker_data]
        # p, = ax.plot(episodes, rewards, marker='.', linestyle='', label='rewards')
        # plots.append(p)
        avg_rwd = []
        last_few = []
        for reward in rewards:
            if len(last_few) >= 40:
                last_few.pop(0)
            last_few.append(reward)
            avg_rwd.append(sum(last_few)/len(last_few))
        p, = ax.plot(steps, avg_rwd, label=f'Epsilon {eps:0.2f}')
        plots.append(p)

    ax.legend(handles=plots, loc='lower right')

    file_name = Path(plot_filename.stem + ' thread by steps' + plot_filename.suffix)
    plt.savefig(out_dir / file_name)


def read_data_and_plot_rewards_old(options):

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

    episode_data = []
    max_reward = 0.0

    for f in csv_files:
        with open(f, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                episode_data.append((int(row['episode']), int(row['steps']), float(row['reward']),
                                     int(row['pid']), float(row['epsilon'])))
                max_reward = max(max_reward, float(row['reward']))

    print(f"max reward from episode data = {max_reward:0.0f}")
    episode_data = sorted(episode_data)

    episode_detail = {}
    total_steps = 0
    for episode, steps, reward, pid, epsilon in episode_data:
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
    cumulative_steps_per_1m = [cs/1000000 for cs in cumulative_steps]
    plot_rewards(cumulative_steps_per_1m, 'steps (millions)', rewards, title, out_dir / file_name)

    plot_filename = Path(options.get('plot_filename', 'async_rewards.png'))
    file_name = plot_filename.stem + ' all episodes' + plot_filename.suffix
    title = options.get('plot_title', 'Asynch Q-Learning')
    plot_all_episodes(episode_data, episode_detail, title, out_dir, Path(file_name))

    plt.close('all')    # matplotlib holds onto all the figures if we don't close them.


def read_data(data_dir):

    # location for input files.
    if not data_dir.exists():
        raise ValueError(f"Could not find data_dir {data_dir}")

    # Gather data from episode detail files.
    p = data_dir.glob('episode-detail*.csv')
    csv_files = [x for x in p if x.is_file()]
    print(csv_files)

    if len(csv_files) == 0:
        print(f"No files found in {data_dir} with pattern 'episode-detail*.csv'.")
        return

    episode_data = []
    max_reward = 0.0

    for f in csv_files:
        with open(f, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                episode_data.append((int(row['episode']), int(row['steps']), float(row['reward']),
                                     int(row['pid']), float(row['epsilon'])))
                max_reward = max(max_reward, float(row['reward']))

    print(f"max reward from episode data = {max_reward:0.0f}")
    episode_data = sorted(episode_data)

    episode_detail = {}
    total_steps = 0
    for episode, steps, reward, pid, epsilon in episode_data:
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

    return {'episodes': episodes, 'rewards': rewards, 'steps': cumulative_steps}


def read_data_and_plot_rewards(options):
    plot_filename = Path(options.get('plot_filename', 'async_rewards.png'))
    # location for output files.
    out_dir = Path(options.get('out_dir'))
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = options.get('data_dirs')

    all_data = {name: read_data(Path(data_dir)) for name, data_dir in data_dirs.items()}

    # file_name = plot_filename.stem + ' by episode' + plot_filename.suffix
    # title = options.get('plot_title', 'Asynch Q-Learning')
    # plot_rewards(episodes, 'episodes', rewards, title, out_dir / file_name)

    # Same again, but by steps

    # plot_filename = Path(options.get('plot_filename', 'async_rewards.png'))
    # file_name = plot_filename.stem + ' by steps' + plot_filename.suffix
    # title = options.get('plot_title', 'Asynch Q-Learning')
    # cumulative_steps_per_100k = [cs/1000000 for cs in cumulative_steps]
    # plot_rewards(cumulative_steps_per_100k, 'steps (millions)', rewards, title, out_dir / file_name)
    #
    # plot_filename = Path(options.get('plot_filename', 'async_rewards.png'))
    # file_name = plot_filename.stem + ' all episodes' + plot_filename.suffix
    # title = options.get('plot_title', 'Asynch Q-Learning')
    # plot_all_episodes(episode_data, title, out_dir, Path(file_name))

    title = options.get('plot_title', 'Asynch Q-Learning')
    file_name = Path(plot_filename.stem + ' by episode' + plot_filename.suffix)
    plot_rewards_by_episode(all_data, title, out_dir / file_name, options.get('include_scatter'))

    plt.close('all')    # matplotlib holds onto all the figures if we don't close them.

if __name__ == '__main__':

    read_data_and_plot_rewards_old(Options({
        'out_dir': "data/async_q_learning/",
        'data_dir': "data/async_q_learning/n_step/",
        'plot_filename': f'async q learning best.png',
        'plot_title': f"Async Q-Learning Breakout 8 workers",
        'include_scatter': False
    }))
    # read_data_and_plot_rewards(Options({
    #     'out_dir': "data/async_q_learning/",
    #     # 'data_dir': "data/async_q_learning/n step no eps decay no lr decay/",
    #     'data_dir': "data/async_q_learning/one step no eps decay no lr decay/",
    #     'data_dirs': {"8-step": "data/async_q_learning/n step no eps decay no lr decay/",
    #                   "1-step": "data/async_q_learning/one step no eps decay no lr decay/"},
    #     'plot_filename': f'1 step vs 8 step.png',
    #     'plot_title': f"Async Q-Learning Breakout 8 workers",
    #     'include_scatter': False
    # }))
