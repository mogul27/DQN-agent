import argparse
from breakout_dqn import run


def main():
    """ Allow options to be entered from the command line when running the python file.

    e.g. The following will display the game
    python run_breakout.py --render=human

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--render", default=None, help="Enter human to see the game played")
    parser.add_argument("--step_size", type=float, default=None, help="Step size when calculating the loss")
    parser.add_argument("--discount_factor", type=float, default=None, help="Discount_factor when calculating the loss")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for Adam optimiser")
    parser.add_argument("--epsilon", type=float, default=None, help="Epsilon for e-greedy policy")
    parser.add_argument("--training_cycles", type=int, default=1, help="Number of training cycles")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per agent")
    parser.add_argument("--load_weights", default=None, help="Load weights from specified file")
    parser.add_argument("--save_weights", default=None, help="Save weights when agent finishes to specified file")


    args = parser.parse_args()

    run(render=args.render,
        training_cycles=args.training_cycles,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        learning_rate=args.learning_rate,
        step_size= args.step_size,
        load_weights=args.load_weights,
        save_weights=args.save_weights
        )


if __name__ == '__main__':
    main()