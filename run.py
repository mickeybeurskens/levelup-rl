""" Run rl-level-up levelup from the command line. """

import gym
import argparse
from levelup.algorithms import *


def get_keyword_arguments(argument_string: str) -> dict:
    key_arg_dict = {}
    if not argument_string:
        return key_arg_dict

    key_arg_strings = argument_string.split(" ")
    for pair in key_arg_strings:
        key, arg = pair.split("=")
        key_arg_dict[key] = arg
    return key_arg_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--show_training', type=bool, default=True)
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--algo', type=str, default='SimplePG')
    parser.add_argument('--algo_params', nargs='+', help="Add extra arguments for the algorithm selected with --algo. "
                                                         "For example: '--algo_params batch_size=1000 "
                                                         "learning_rate=0.001'")
    args = parser.parse_args()

    if args.algo not in globals():
        raise ValueError("Algorithm " + args.algo + " not found")
    model: BaseAlgorithm = globals()[args.algo]()
    model.train(gym.make(args.env), args.epochs, args.episodes,
                args.show_training, args.exp_name,
                **get_keyword_arguments(args.algo_params))
