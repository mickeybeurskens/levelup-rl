""" Warming up with a custom pytorch nn class
Based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

Read up on theory at http://neuralnetworksanddeeplearning.com/
"""
import torch


class FullyConNet(torch.nn.Module):
    def __init__(self, dim_hid, dim_in, dim_out):
        super(FullyConNet, self).__init__()
        self.layer_in = torch.nn.Linear(dim_in, dim_hid)
        self.relu = torch.nn.ReLU()
        self.layer_out = torch.nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        h_relu = self.relu(self.layer_in(x))
        output_pred = self.layer_out(h_relu)
        return output_pred


if __name__ == "__main__":
    batch_dim, dim_hid, dim_in, dim_out = 64, 100, 1000, 10

    net_data_input: torch.Tensor = torch.randn(batch_dim, dim_in)
    net_data_output: torch.Tensor = torch.randn(batch_dim, dim_out)

    model = FullyConNet(dim_hid, dim_in, dim_out)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(500):
        output_prediction: torch.Tensor = model(net_data_input)
        loss: torch.Tensor = loss_fn(output_prediction, net_data_output)
        if t % 100 == 99:
            print(t, loss.item())

        # Clear gradient buffers in tensors
        optimizer.zero_grad()
        # Compute new gradients w.r.t. loss and store in tensors
        loss.backward()
        # Use gradients and optimizer to compute new model parameters
        optimizer.step()
