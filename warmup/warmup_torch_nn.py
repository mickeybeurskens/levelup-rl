""" Warming up with a pytorch nn classes
Based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

Read up on theory at http://neuralnetworksanddeeplearning.com/
"""

import torch

if __name__ == "__main__":
    batch_dim, dim_hid, dim_in, dim_out = 64, 100, 1000, 10

    net_data_input: torch.Tensor = torch.randn(batch_dim, dim_in)
    net_data_output: torch.Tensor = torch.randn(batch_dim, dim_out)

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_hid, dim_out),
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    for t in range(500):
        output_prediction: torch.Tensor = model(net_data_input)
        loss: torch.Tensor = loss_fn(output_prediction, net_data_output)
        if t % 100 == 99:
            print(t, loss.item())

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
