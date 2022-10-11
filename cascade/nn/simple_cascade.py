import torch
from torch import nn
from .cascade_nn import CascadeNN
from .cascade_q import CascadeQ
from cascade.utils import clone_lin_model

# class MLP(torch.nn.Module):
#     def __init__(self, sizes, non_lin):
#         super().__init__()
#         ops = []
#         self.non_lin = non_lin
#         self.sizes = sizes
#         for si, so in zip(sizes[:-1], sizes[1:]):
#             ops.append(torch.nn.Linear(si, so))
#             ops.append(non_lin)
#         self.f = torch.nn.Sequential(*ops[:-1])

#     def forward(self, x):
#         return self.f(x)



class ConCatLayer(torch.nn.Module):
    def __init__(self, net, concat_dim = 1):
        super().__init__()
        self.net = net
        self.concat_dim = concat_dim
    def forward(self, x):
        return torch.cat([x, self.net(x)], dim = self.concat_dim)

class Simple_Cascade(CascadeNN):
    def __init__(self, dim_input, dim_output, hidden_layer_sizes, non_lin = torch.nn.ReLU()):
        super().__init__(dim_input, dim_output)
        self.nb_hidden = 0
        self.cascade_neurone_list = []
        linear_layer_list = []
        sizes = [dim_input] + hidden_layer_sizes
        for si, so in zip(sizes[:-1], sizes[1:]):
            linear_layer_list.append(torch.nn.Linear(si, so))
            linear_layer_list.append(non_lin)
        self.nb_hidden += so
        self.cascade_neurone_list = [ConCatLayer(nn.Sequential(*linear_layer_list))]
        self.cascade = nn.Sequential(*self.cascade_neurone_list)
        self.output = nn.Linear(dim_input + self.nb_hidden, dim_output)
        self.output.weight.data[:] = 0.
        self.output.bias.data[:] = 0.


class SimpleCascadeQ(CascadeQ, Simple_Cascade):
    def __init__(self, dim_input, dim_output, hidden_layer_sizes, non_lin = torch.nn.ReLU(), **kwargs):
        kwargs["hidden_layer_sizes"] = hidden_layer_sizes
        kwargs["non_lin"] = non_lin
        super().__init__(dim_input=dim_input, dim_output=dim_output, **kwargs)
    
    def sync_outputs(self, qfunc, output):
        #assumes that qfunc and output have the same shape as input models
        self.output = clone_lin_model(output)
        self.qfunc = clone_lin_model(qfunc)
