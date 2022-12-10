import argparse
from breakout_dqn import run


def main():
    """ Allow options to be entered from the command line when running the python file.

    e.g. The following will display the game
    python run_breakout.py --render=human

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--render", default=None, help="Enter human to see the game played")
    parser.add_argument("--discount_factor", type=float, default=None, help="Discount_factor when calculating the loss")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for Adam optimiser")
    parser.add_argument("--epsilon", type=float, default=None, help="Epsilon for e-greedy policy")
    parser.add_argument("--epsilon_decay_span", type=float, default=None, help="Epsilon decay for e-greedy policy")
    parser.add_argument("--epsilon_min", type=float, default=None, help="Epsilon min for e-greedy policy")
    parser.add_argument("--training_cycles", type=int, default=1, help="Number of training cycles")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per agent")
    parser.add_argument("--load_weights", default=None, help="Load weights from specified file")
    parser.add_argument("--save_weights", default=None, help="Save weights when agent finishes to specified file")
    parser.add_argument("--work_dir", default=None, help="Working dir to save files to")


    args = parser.parse_args()

    run(render=args.render,
        training_cycles=args.training_cycles,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        epsilon_decay_span=args.epsilon_decay_span,
        epsilon_min=args.epsilon_min,
        learning_rate=args.learning_rate,
        load_weights=args.load_weights,
        save_weights=args.save_weights,
    work_dir=args.work_dir
        )


if __name__ == '__main__':
    main()