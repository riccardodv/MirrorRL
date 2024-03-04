import torch
import torch.nn as nn
from cascade.utils import clone_lin_model





class CascadeNeuroneBN(nn.Module):
    def __init__(self, nb_in, nb_out, non_linearity=nn.ReLU()):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nb_in, out_features=nb_out), non_linearity)
        self.bn = nn.BatchNorm1d(nb_out)
        self.nb_in = nb_in

    def forward(self, x):
        return self.bn(self.f(x))


class CascadeNNBN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.nb_hidden = 0
        self.cascade_neurone_list = []
        self.output = nn.Linear(dim_input + self.nb_hidden, dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.

    def __call__(self, x):
        return self.output(self.get_features(x))

    def get_features(self, x):
        features = torch.zeros(x.shape[0], self.nb_hidden + x.shape[1])
        nb_filled_features = x.shape[1]
        features[:, :nb_filled_features] = x
        for casc in self.cascade_neurone_list:
            nb_out = casc.f[0].out_features
            features[:, nb_filled_features:nb_filled_features+nb_out] = casc(features[:, :nb_filled_features])
            nb_filled_features += nb_out
        return features

    def add_n_neurones(self, n_neurones=1, non_linearity=nn.ReLU()):
        new_neurone = CascadeNeuroneBN(self.dim_input + self.nb_hidden, nb_out=n_neurones, non_linearity=non_linearity)
        self.cascade_neurone_list.append(new_neurone)
        self.nb_hidden += n_neurones
        old_out = clone_lin_model(self.output)
        self.output = nn.Linear(self.dim_input + self.nb_hidden, self.dim_output)
        self.output.weight.data[:] = 0.
        self.output.weight.data[:, :-n_neurones] = old_out.weight
        self.output.bias.data[:] = old_out.bias

    def forward_from_frozen_features(self, feat):
        return self.output(torch.cat([feat, self.last_neurone()(feat)], dim=1))

    def last_neurone(self):
        return self.cascade_neurone_list[-1]

    def train(self, train_mode):
        if self.nb_hidden > 0:
            self.last_neurone().train(train_mode)

    def parameters(self):
        return [*self.last_neurone().parameters(), *self.output.parameters()]
