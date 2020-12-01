import gym
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """ Define interface for reinforcement learning levelup. """

    @abstractmethod
    def train(self, env_fn: gym.Env, epochs: int, episodes: int,
              show_training: bool, exp_name: str, **kwargs):
        """
        Train an algorithm in a OpenAI gym compatible environment.

        :param env_fn: OpenAI Gym environment instance.
        :param epochs: Number of training epochs.
        :param episodes: Number of training episodes.
        :param show_training: Visualize a training run each epoch.
        :param exp_name: Current experiment name.
        :param kwargs: Algorithm specific parameters.
        :return:
        """
        pass
