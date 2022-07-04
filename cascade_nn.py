import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from msc_tools import clone_lin_model


class CascadeNeurone(nn.Module):
    def __init__(self, nb_in, nb_out, dim_data_input, non_linearity=nn.ReLU()):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nb_in, out_features=nb_out), non_linearity)
        self.in_index = [k for k in range(dim_data_input)] + [k for k in range(dim_data_input - nb_in, 0, 1)]  # data input index + selected feature index
        self.nb_in = nb_in

    def forward(self, x, stack=True):
        in_x = x
        if x.shape[1] > self.nb_in:
            in_x = x[:, self.in_index]
        if stack:
            # return torch.column_stack([x, self.f(in_x)])
            return torch.cat([x, self.f(in_x)], dim=1)
        else:
            return self.f(in_x)


class CascadeNN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.nb_hidden = 0
        self.cascade_neurone_list = []
        self.cascade = nn.Sequential(*self.cascade_neurone_list)
        self.output = nn.Linear(dim_input + self.nb_hidden, dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.

    def forward(self, x, stack=True):
        return self.output(self.get_features(x, stack))

    def get_features(self, x, stack=True):
        if stack:
            return self.cascade(x)
        else:
            features = torch.zeros(x.shape[0], self.nb_hidden + x.shape[1])
            features[:, :x.shape[1]] = x
            for casc in self.cascade:
                nb_in = casc.f[0].in_features
                nb_out = casc.f[0].out_features
                features[:, nb_in:nb_in+nb_out] = casc(features[:, :nb_in], stack=False)
            return features

    def add_n_neurones(self, all_features, nb_inputs, n_neurones=1, non_linearity=nn.ReLU(), init_from_old=False):
        assert nb_inputs >= self.dim_input, f'nb_inputs must at least be larger than data dimensionality {self.dim_input}'
        nb_inputs = min(nb_inputs, all_features.shape[1])
        new_neurone = CascadeNeurone(nb_inputs, nb_out=n_neurones, dim_data_input=self.dim_input, non_linearity=non_linearity)
        all_features = all_features[:, new_neurone.in_index]
        new_neurone.f[0].weight.data /= 2 * (new_neurone.f[0].weight @ all_features.t().cpu()).abs().max(dim=1)[0].unsqueeze(1)
        new_neurone.f[0].bias.data = -torch.mean(new_neurone.f[0].weight @ all_features.t().cpu(), dim=1).detach() # I DONT UNDERSTAND; could be related to paper?
        self.cascade_neurone_list.append(new_neurone)
        self.nb_hidden += n_neurones
        self.cascade = nn.Sequential(*self.cascade_neurone_list)
        if init_from_old:
            old_out = clone_lin_model(self.output)
        self.output = nn.Linear(self.dim_input + self.nb_hidden, self.dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.
        if init_from_old:
            self.output.weight.data[:, :-n_neurones] = old_out.weight
            self.output.bias.data[:] = old_out.bias

    def forward_from_old_cascade_features(self, feat):
        return self.output(self.get_features_from_old_cascade_features(feat))

    def get_features_from_old_cascade_features(self, feat):
        return self.cascade_neurone_list[-1](feat)

    def merge_with_old_weight_n_bias(self, old_weight, old_bias): #shouldn't mean be better, though the sum is probably coming from the paper
        self.output.weight.data[:, :old_weight.shape[1]] += old_weight
        self.output.bias.data += old_bias

    def parameters_last_only(self):
        return [*self.cascade_neurone_list[-1].parameters(), *self.output.parameters()]


class CosDataset(TensorDataset):
    def __init__(self, nb_points):
        self.nb_points = nb_points
        self.x = 10 * torch.rand(nb_points, 1) - 5
        self.y = torch.cos(self.x) + 1e-1 * torch.randn(nb_points, 1)
        super().__init__(self.x, self.y)


def test_reg_inc(nb_points):
    data = CosDataset(nb_points)
    model = CascadeNN(dim_input=1, dim_output=1)
    loss = nn.MSELoss()
    min_nsteps = 50000
    nsteps = 0
    add_every_n_epoc = 200
    nb_added_neurones = 2
    connect_last_k_features = 3 * nb_added_neurones
    dim_data_input = 1
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
            model.add_n_neurones(features, nb_inputs=connect_last_k_features + dim_data_input, n_neurones=nb_added_neurones)  # add just one neuron
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
    test_reg_inc(nb_points=128)
