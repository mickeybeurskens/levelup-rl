""" Warming up with pytorch autograd
Based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

Read up on theory at http://neuralnetworksanddeeplearning.com/
"""

import torch

if __name__ == "__main__":
    dtype = torch.float
    device = torch.device("cuda:0")

    batch_size, dim_in, hid_dim, dim_out = 64, 1000, 100, 10

    input_data: torch.Tensor = torch.randn(batch_size, dim_in,
                                           device=device, dtype=dtype)
    output_data: torch.Tensor = torch.randn(batch_size, dim_out,
                                            device=device, dtype=dtype)

    net_weight_1: torch.Tensor = torch.randn(dim_in, hid_dim, device=device,
                                             dtype=dtype, requires_grad=True)
    net_weight_2: torch.Tensor = torch.randn(hid_dim, dim_out, device=device,
                                             dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        #  mm stands for matrix multiplication
        net_prediction = input_data.mm(net_weight_1).clamp(min=0).mm(net_weight_2)

        loss: torch.Tensor = (net_prediction - output_data).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())
        loss.backward()

        # Manual update requires disabling auto gradient computation
        with torch.no_grad():
            net_weight_1 -= learning_rate * net_weight_1.grad
            net_weight_2 -= learning_rate * net_weight_2.grad

            # Reset gradients manually after computation
            net_weight_1.grad.zero_()
            net_weight_2.grad.zero_()
