import torch
from abc import ABCMeta
from typing import List


class FullyConnectedNetwork(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, in_size, out_size, hidden_sizes: List[int]):
        super(FullyConnectedNetwork, self).__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.rectified_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        if len(hidden_sizes) > 0:
            prev_size = in_size
            for size in hidden_sizes:
                self.layers.append(torch.nn.Linear(prev_size, size))
                self.rectified_layers.append(torch.nn.ReLU())
                prev_size = size
            self.output_layer = torch.nn.Linear(prev_size, out_size)
        else:
            raise ValueError("Not enough hidden layers")

    def forward(self, layout):
        for layer, relu in zip(self.layers, self.rectified_layers):
            layout = relu(layer(layout))
        return self.output_layer(layout)

    def get_policy(self, x):
        prediction = self.forward(x)
        return torch.distributions.Categorical(logits=prediction)

    def get_action(self, x):
        return self.get_policy(x).sample().item()

    def get_loss(self, observations, actions, weights):
        """ For the right data (i.e. action, observation, reward/weights pairs)
        this is equal to the loss.

        Maximizing the reward gradient means minimizing loss because of the
        minus sign. In essence the policy gradient is the direction of
        maximal reward when updated the network parameters. Flipping this
        function gives a loss. The policy gradient is thus used as a direction
        for network parameter updates. Weights are based on the total reward a
        that training episode.

        For more information take a look at the policy gradient estimate g at
        https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
        """
        delta_log_p = self.get_policy(observations).log_prob(actions)
        return -(delta_log_p * weights).mean()
