import gym
import torch
import numpy as np
from typing import List, Optional
from levelup.logging.logger import Logger
from levelup.algorithms.models import FullyConnectedNetwork
from levelup.algorithms.base_algorithm import BaseAlgorithm


class SimplePG(BaseAlgorithm):
    def __init__(self):
        self.logger = Logger()
        self.name = "spg_default"
        self.print = False

    def train(self, env_fn: gym.Env, epochs: int, episodes: int,
              show_training: bool, exp_name: str, batch_size: int = 10000,
              hidden_sizes: List[int] = None, learning_rate: float = None,
              print_training_data: bool = False):
        if hidden_sizes is None:
            hidden_sizes = [64, 16]
        if learning_rate is None:
            learning_rate = 1e-4
        self.name = exp_name
        self.print = print_training_data

        model = self._get_model(env_fn, hidden_sizes)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self._train_epoch(epoch, env_fn, episodes, show_training,
                              batch_size, optimizer, model)
        env_fn.close()

    def _print_training_info(self, epoch, episode, ep_rew):
        if self.print:
            if episode % 100 == 0 or episode == 0:
                print("Epoch " + str(epoch) + ", episode " + str(episode) +
                      ", reward: " + str(np.sum(ep_rew)))

    @staticmethod
    def _reset_replay_memory():
        return {"observations": [], "actions": [], "rewards": [],
                "weights": []}

    @staticmethod
    def _update_network_params(optimizer, model, mem):
        optimizer.zero_grad()
        batch_loss = model.get_loss(torch.as_tensor(mem["observations"],
                                                    dtype=torch.float32),
                                    torch.as_tensor(mem["actions"],
                                                    dtype=torch.float32),
                                    torch.as_tensor(mem["weights"],
                                                    dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()

    @staticmethod
    def _get_model(env_fn, hidden_sizes):
        input_size = env_fn.observation_space.shape[0]
        output_size = env_fn.action_space.n
        return FullyConnectedNetwork(input_size, output_size, hidden_sizes)

    def _train_epoch(self, epoch, env_fn, episodes, show_training, batch_size,
                     optimizer, model):
        finished_rendering_this_epoch = False
        mem = self._reset_replay_memory()
        # A round of training happens within an epoch, with data from episodes
        for episode in range(episodes):
            obs = env_fn.reset()
            ep_rew = []
            while True:
                if not finished_rendering_this_epoch and show_training:
                    env_fn.render()
                mem["observations"].append(obs.copy())
                action = model.get_action(torch.as_tensor(obs, dtype=torch.float32))
                obs, rew, done, info = env_fn.step(action)

                ep_rew.append(rew)
                mem["actions"].append(action)
                if done:
                    # Batch weights can only be assigned when success is known
                    ep_ret, ep_len = sum(ep_rew), len(ep_rew)
                    mem["weights"].extend([ep_ret] * ep_len)
                    mem["rewards"].extend(ep_rew)
                    finished_rendering_this_epoch = True
                    self._print_training_info(epoch, episode, ep_rew)
                    self.logger.log(self.name, epoch, episode, ep_ret)
                    self.logger.write(self.name + '.csv')
                    break

            if len(mem["actions"]) > batch_size or episode == range(episodes)[-1]:
                self._update_network_params(optimizer, model, mem)
                mem = self._reset_replay_memory()
