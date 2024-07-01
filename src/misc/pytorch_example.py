#! /usr/bin/env python3
"""
Get to grips with pyTorch.

Learn it curve-fitting as nonsense example.

Based on:
- https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def ref_fun(x):
    return -4 * x**2 + 30 * x + 2
    # return 2 * x**2 + 4 * x  # much simpler


# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = torch.nn.Flatten()
        n_hidden = 5
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1, n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            # Extra 1 #
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            # Extra 2 #
            # torch.nn.Linear(n_hidden, n_hidden),
            # torch.nn.LeakyReLU(),
            # torch.nn.Dropout(0.1),
            # Final
            torch.nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        result = self.linear_relu_stack(x)
        return result


def main():
    "Run it all."

    torch.manual_seed(2)
    np.random.seed(2)

    # ########## Create some data ##########
    x_lims = -10, 15
    x_ref = np.expand_dims(np.arange(x_lims[0], x_lims[1], 0.1), axis=1)
    y_ref = np.array([ref_fun(x) for x in x_ref])
    # plt.plot(x_ref, y_ref)
    # plt.grid()
    # plt.show()
    # print(x_ref.shape, y_ref.shape)

    # ########## Do a model ##########
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    print(model)

    # ########## Train it ##########
    loss_fn = torch.nn.L1Loss()  # torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train(True)

    num_data = 500
    num_trials = 5000
    # num_data = 50
    # num_trials = 50000

    stats_i = []
    stats_l = []
    for i_trial in range(num_trials):
        # Generate random datas
        x_train = np.random.uniform(x_lims[0], x_lims[1], size=(num_data, 1))
        y_train = np.array([ref_fun(x) for x in x_train])

        x_ten = torch.Tensor(x_train.astype("float32"))
        y_ten = torch.Tensor(y_train.astype("float32"))

        # Compute prediction error
        pred = model(x_ten)
        loss = loss_fn(pred, y_ten)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i_trial % 100 == 0:
            loss, current = loss.item(), (i_trial + 1) * len(x_ten)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_data:>5d}]")
            stats_i.append(i_trial)
            stats_l.append(loss)

    model.eval()

    plt.plot(stats_i, stats_l)
    plt.show()

    # ########## Evaluate ##########
    x_test = torch.tensor(x_ref.astype("float32"))
    y_test = model(x_test)
    y_result = y_test.detach().numpy()
    plt.plot(x_ref, y_ref, "-b")
    plt.plot(x_ref, y_result, "-r")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
