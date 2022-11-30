import torch
import torch.nn as nn
from cascade.utils import clone_lin_model

class FeedForward(torch.nn.Module):
    def __init__(self, in_size, out_size, layer_sizes, activation='RELU'):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes)):
            if i == 0:
                self.layers.append(torch.nn.Linear(in_size, layer_sizes[i]))
            else:
                self.layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.output = torch.nn.Linear(layer_sizes[-1], out_size)
        if activation == 'RELU':
            self.activation = torch.nn.ReLU()
        elif activation == 'SIGMOID':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'TANH':
            self.activation = torch.nn.Tanh()
        elif activation == 'IDENTITY':
            self.activation = torch.nn.Identity()
        else:
            raise ValueError('Activation not supported')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.output(x)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def copy_model(self, external_nn):
        for i in range(len(self.layers)):
            self.layers[i] = clone_lin_model(external_nn.layers[i])
        self.output = clone_lin_model(external_nn.output)
        self.activation = external_nn.activation


class EnsembleNN(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def add_model(self, model):
        self.models.append(model)
    
    def forward(self, x):
        s = 0
        for m in self.models:
            s = s + m(x)
        return s

    def forward_boosting(self, x):
        s = 0
        for i in range(len(self.models)):
            coef = len(self.models) - i
            s = s + coef*self.models[i](x)
        return s

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
    def __init__(self, dim_input, dim_output, **kwargs):
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

    def get_features_from_old_cascade_features(self, feat, stack=True):
        return self.cascade_neurone_list[-1](feat, stack=stack)

    def merge_with_old_weight_n_bias(self, old_weight, old_bias): #shouldn't mean be better, though the sum is probably coming from the paper
        self.output.weight.data[:, :old_weight.shape[1]] += old_weight
        self.output.bias.data += old_bias

    def parameters_last_only(self):
        return [*self.cascade_neurone_list[-1].parameters(), *self.output.parameters()]

    def feat_last_only(self):
        return [*self.cascade_neurone_list[-1].parameters()]


        

