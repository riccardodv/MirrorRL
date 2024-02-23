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


class TargetCascadeNN2(nn.Module):
    def __init__(self, dim_input, dim_output, nb_hidden, added_nb_hidden, non_linearity = nn.ReLU()):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.nb_hidden = nb_hidden + added_nb_hidden
        self.cascade_neurone_list = []
        # if added_nb_hidden > 0:
        #     self.cascade_neurone_list.append(CascadeNeurone(nb_in = dim_input+nb_hidden, nb_out=added_nb_hidden, dim_data_input=dim_input, non_linearity= non_linearity))
        self.cascade = CascadeNeurone(nb_in = dim_input+nb_hidden, nb_out=added_nb_hidden, dim_data_input=dim_input, non_linearity= non_linearity)
        self.output = nn.Linear(self.nb_hidden, dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.

    
    def forward(self, x, stack=True):
        return self.output(self.get_features(x, stack)[..., -self.nb_hidden:])

    def get_features(self, old_feat, stack=True):
        if stack:
            return self.cascade(old_feat)
        else:
            features = torch.zeros(old_feat.shape[0], self.nb_hidden + old_feat.shape[1])
            features[:, :old_feat.shape[1]] = old_feat
            for casc in self.cascade:
                nb_in = casc.f[0].in_features
                nb_out = casc.f[0].out_features
                features[:, nb_in:nb_in+nb_out] = casc(features[:, :nb_in], stack=False)
            return features
        
    def load_weight(self, cascade_neurone, output_l):
        self.output = clone_lin_model(output_l)
        self.cascade.f[0].weight = cascade_neurone.f[0].weight.detach().clone()
        self.cascade.f[0].bias = cascade_neurone.f[0].bias.detach().clone()

    def alpha_sync(self, cascade_neurone, output_l, alpha):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0

        self.output.weight = alpha*self.output.weight + (1-alpha)*output_l.weight.detach()
        self.output.bias = alpha*self.output.bias + (1-alpha)*output_l.bias.detach()

        self.cascade.f[0].weight = alpha*self.cascade.f[0].weight + (1-alpha)*cascade_neurone.f[0].weight.detach()
        self.cascade.f[0].bias = alpha*self.cascade.f[0].bias + (1-alpha)*cascade_neurone.f[0].bias.detach()
 

class CascadeNN2(nn.Module):
    def __init__(self, dim_input, dim_output, init_nb_hidden = 1, non_linearity = nn.ReLU(), **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.nb_hidden = init_nb_hidden
        self.cascade_neurone_list = []
        if init_nb_hidden > 0:
            self.cascade_neurone_list.append(CascadeNeurone(nb_in = dim_input, nb_out=init_nb_hidden, dim_data_input=dim_input, non_linearity= non_linearity))
        self.cascade = nn.Sequential(*self.cascade_neurone_list)
        self.output = nn.Linear(self.nb_hidden, dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.

    def forward(self, x, stack=True):
        return self.output(self.get_features(x, stack)[..., -self.nb_hidden:])

    def get_features(self, x, stack=True):
        if stack:
            return self.cascade(x)
        # else:
        #     features = torch.zeros(x.shape[0], self.nb_hidden)
        #     st_in = 0
        #     # features[:, :x.shape[1]] = x
        #     for casc in self.cascade:
        #         # nb_in = casc.f[0].in_features
        #         nb_out = casc.f[0].out_features
        #         output = casc(x, stack=False)
        #         features[:, st_in:st_in+nb_out] = output
        #         x = torch.cat([x, output], dim = -1)
        #         st_in +=nb_out
        #     return features
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
        self.output = nn.Linear(self.nb_hidden, self.dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.
        if init_from_old:
            self.output.weight.data[:, :-n_neurones] = old_out.weight
            self.output.bias.data[:] = old_out.bias

    def forward_from_old_cascade_features(self, feat):
        return self.output(self.get_features_from_old_cascade_features(feat)[..., -self.nb_hidden:])

    def get_features_from_old_cascade_features_cropped(self, feat):
        """Get final feature from the last cascade neurone, but only the ones that are not the input features"""
        return self.get_features_from_old_cascade_features(feat)[..., -self.nb_hidden:]
    
    def forward_from_features_without_grad(self, feat):
        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False
        res = self.output(feat)
        self.output.weight.requires_grad = True
        self.output.bias.requires_grad = True
        return res


    def get_features_from_old_cascade_features(self, feat, stack=True):
        return self.cascade_neurone_list[-1](feat, stack=stack)

    def merge_with_old_weight_n_bias(self, old_weight, old_bias): #shouldn't mean be better, though the sum is probably coming from the paper
        self.output.weight.data[:, :old_weight.shape[1]] += old_weight
        self.output.bias.data += old_bias

    def parameters_last_only(self):
        return [*self.cascade_neurone_list[-1].parameters(), *self.output.parameters()]

    def feat_last_only(self):
        return [*self.cascade_neurone_list[-1].parameters()]

        

