""" Warming up with a custom pytorch autograd
Based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

Read up on theory at http://neuralnetworksanddeeplearning.com/
"""

import torch


class MyReLU(torch.autograd.Function):
    """ A linear rectified unit auto gradient class. """

    @staticmethod
    def forward(ctx, tensor_in):
        """ Take a tensor and return the corresponding output. ctx is a
        context object that can contain arbitrary information the using
        ctx.save_for_backward method
        """
        ctx.save_for_backward(tensor_in)
        return tensor_in.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward receives a tensor containing the gradient of
        the loss function w.r.t. the output. It returns the gradient
        of the loss function w.r.t. the input."""
        tensor_in, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[tensor_in < 0] = 0
        return grad_input


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
        # Applying the auto gradient happens with the apply method
        relu = MyReLU.apply

        #  mm stands for matrix multiplication
        net_prediction = relu(input_data.mm(net_weight_1)).mm(net_weight_2)

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
