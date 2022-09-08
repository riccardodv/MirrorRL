import torch
import torch.nn as nn


def clone_lin_model(m):
    ret = nn.Linear(m.in_features, m.out_features)
    ret.weight.data = m.weight.detach().clone()
    ret.bias.data = m.bias.detach().clone()
    return ret


def norm_squared_lin(m):
    return m.weight.pow(2).sum() + m.bias.pow(2).sum()


def stable_kl_div(old_probs, new_probs, epsilon=1e-12):
    new_probs = new_probs + epsilon
    old_probs = old_probs + epsilon
    kl = new_probs*torch.log(new_probs) - new_probs*torch.log(old_probs)
    return kl