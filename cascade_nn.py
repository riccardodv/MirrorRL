import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


class CascadeNeurone(nn.Module):
    def __init__(self, dim_input, dim_output, non_linearity=nn.ReLU()):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim_input, out_features=dim_output), non_linearity)

    def forward(self, x):
        return torch.column_stack([x, self.f(x)])


class CascadeNN(nn.Module):
    def __init__(self, dim_input, dim_output, init_nb_hidden=0):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.nb_hidden = init_nb_hidden
        self.cascade_neurone_list = []
        for k in range(init_nb_hidden):
            self.cascade_neurone_list.append(CascadeNeurone(dim_input + k, dim_output=1))
        self.cascade = nn.Sequential(*self.cascade_neurone_list)
        self.output = nn.Linear(dim_input + self.nb_hidden, dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.

    def forward(self, x):
        return self.output(self.get_features(x))

    def get_features(self, x):
        return self.cascade(x)

    def add_n_neurones(self, features, n=1, non_linearity=nn.ReLU()):
        new_neurone = CascadeNeurone(self.dim_input + self.nb_hidden, dim_output=n, non_linearity=non_linearity)
        new_neurone.f[0].weight.data /= 2 * (new_neurone.f[0].weight @ features.t()).abs().max(dim=1)[0].unsqueeze(1)
        new_neurone.f[0].bias.data = -torch.mean(new_neurone.f[0].weight @ features.t(), dim=1).detach() # I DONT UNDERSTAND; could be related to paper?
        self.cascade_neurone_list.append(new_neurone)
        self.nb_hidden += n
        self.cascade = nn.Sequential(*self.cascade_neurone_list)
        self.output = nn.Linear(self.dim_input + self.nb_hidden, self.dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.

    def forward_from_old_cascade_features(self, feat):
        return self.output(self.get_features_from_old_cascade_features(feat))

    def get_features_from_old_cascade_features(self, feat):
        return self.cascade_neurone_list[-1](feat)

    def merge_with_old_weight_n_bias(self, old_weight, old_bias): #shouldn't mean be better, though the sum is probably coming from the paper
        self.output.weight.data[:, :old_weight.shape[1]] += old_weight
        self.output.bias.data += old_bias


class CosDataset(TensorDataset):
    def __init__(self, nb_points):
        self.nb_points = nb_points
        self.x = 10 * torch.rand(nb_points, 1) - 5
        self.y = torch.cos(self.x) + 1e-1 * torch.randn(nb_points, 1)
        super().__init__(self.x, self.y)


def test_reg_full(nb_points):
    data = CosDataset(nb_points)

    data_loader = DataLoader(data, batch_size=16, shuffle=True, drop_last=True)
    model = CascadeNN(dim_input=1, dim_output=1, init_nb_hidden=32)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    min_nsteps = 50000
    nsteps = 0

    while nsteps < min_nsteps:
        train_error = []
        for x, y in data_loader:
            optim.zero_grad()
            train_error.append(loss(model(x), y))
            train_error[-1].backward()
            optim.step()
            nsteps += 1
        print(f'nsteps {nsteps} training error {sum(train_error).item()/len(train_error)}')

    plt.figure()
    plt.plot(data.x, data.y, 'x')
    x = torch.linspace(-5, 5, steps=1000)[:, None]
    with torch.no_grad():
        y = model(x)
    plt.plot(x, y, '-')
    plt.show()


def test_reg_inc(nb_points):
    data = CosDataset(nb_points)
    model = CascadeNN(dim_input=1, dim_output=1, init_nb_hidden=0)
    loss = nn.MSELoss()
    min_nsteps = 50000
    nsteps = 0
    add_every_n_epoc = 200
    epoch = 0
    old_weight, old_bias = None, None
    while nsteps < min_nsteps:
        if epoch % add_every_n_epoc == 0:
            # Merging old model with new if any
            if old_weight is not None:
                model.merge_with_old_weight_n_bias(old_weight, old_bias)
            with torch.no_grad():
                train_error = loss(model(data.x), data.y).item()
            print(f'nsteps {nsteps} nneurones {model.nb_hidden} training error {train_error}')

            features = model.get_features(data.x).detach()
            residuals = data.y - model(data.x).detach()
            old_weight, old_bias = model.output.weight.detach().clone(), model.output.bias.detach().clone()
            data_loader = DataLoader(TensorDataset(features, residuals), batch_size=16, shuffle=True, drop_last=True)  # dataset of features and residuals, the idea from some paper?
            model.add_n_neurones(features)  # add just one neuron
            optim = torch.optim.Adam([*model.cascade_neurone_list[-1].parameters(), *model.output.parameters()], lr=1e-3)
        for feat, residual in data_loader:
            optim.zero_grad()
            loss(model.forward_from_old_cascade_features(feat), residual).backward()
            optim.step()
            nsteps += 1
        epoch += 1
    print(model)

    model.merge_with_old_weight_n_bias(old_weight, old_bias)
    plt.figure()
    plt.plot(data.x, data.y, 'x')
    x = torch.linspace(-5, 5, steps=1000)[:, None]
    with torch.no_grad():
        y = model(x)
    plt.plot(x, y, '-')
    plt.show()


if __name__ == '__main__':
    # test_reg_full(nb_points=128)
    test_reg_inc(nb_points=128)
