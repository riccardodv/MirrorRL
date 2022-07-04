import numpy as np
import torch
from rl_tools import Sampler
from pendulum_discrete import PendulumDiscrete
import matplotlib.pyplot as plt
from cascade_nn import CascadeNN
from torch.utils.data import DataLoader, TensorDataset
from lstdq_torch import lstd_q


def rwds_finished_rollouts(samples):
    returns = []
    ret = 0
    for d, r in zip(samples['done'], samples['rwd']):
        ret += r.item()
        if d:
            returns.append(ret)
            ret = 0
    return returns


def deterministic_policy(obs, cascade_q):
    with torch.no_grad():
        return torch.argmax(cascade_q(obs[None, :])).cpu().numpy()


pol = torch.load('models/cascade_qfunc_DiscretePendulum_iter15.pt')
env = PendulumDiscrete()
env_sampler = Sampler(env)
nb_samp = 10048
device = torch.device('cpu')
rolls_learn = env_sampler.rollouts(policy=lambda x: deterministic_policy(x, pol), min_trans=nb_samp, max_trans=nb_samp, device=device)
non_linearity = torch.nn.ReLU()
use_lstd = True
# rolls_test = env_sampler.rollouts(policy=lambda x: deterministic_policy(x, pol),
#                              min_trans=nb_samp, max_trans=nb_samp, device=device)
# returns = rwds_finished_rollouts(rolls_learn)
# print(returns)
# plt.plot(returns, 'x')
# plt.show()

rwd, ter, obs, nobs, act, nact = rolls_learn['rwd'], rolls_learn['terminal'].float(), rolls_learn['obs'],\
                                 rolls_learn['nobs'], rolls_learn['act'].long(), rolls_learn['nact'].long()

gamma = .99
neurone_per_iter = 10
qfunc = CascadeNN(env.get_dim_obs(), env.get_nb_act())
nb_iter = 150
epoch_per_iter = 30
batch_size = 64
lr = 1e-3
loss_fct = torch.nn.MSELoss()
msbes_train = []

for it in range(nb_iter):
    # computing Q target
    with torch.no_grad():
        obs_features = qfunc.get_features(obs)
        nobs_features = qfunc.get_features(nobs)
    qfunc.add_n_neurones(obs_features, torch.inf, neurone_per_iter, non_linearity=non_linearity, init_from_old=True)
    if use_lstd:
        phis = qfunc.get_features(obs)
        nphis = qfunc.get_features(nobs)
        bias, weight = lstd_q(phis, act, rwd, nphis, nact, 1 - ter, gamma, env.get_nb_act(), add_bias=True)
        qfunc.output.bias.data, qfunc.output.weight.data = bias, weight

        # logging
        qtarg = rwd + gamma * (1 - ter) * qfunc.output(nphis).gather(dim=1, index=nact)
        qvals = qfunc.output(phis)
        msbes_train.append(loss_fct(qvals, qtarg).item())
        print(f'iter {it}: msbe {msbes_train[-1]:5.3f}')

    else:
        for e in range(epoch_per_iter):
            with torch.no_grad():
                qtarg = rolls_learn['rwd'] + gamma * (1 - ter) * qfunc.forward_from_old_cascade_features(nobs_features).gather(dim=1, index=nact)
                # logging
                qvals = qfunc.forward_from_old_cascade_features(obs_features)
                msbes_train.append(loss_fct(qvals, qtarg).item())
                print(f'iter {it}, epoch {e}: msbe {msbes_train[-1]:5.3f}')

            data_loader = DataLoader(TensorDataset(obs_features, act, qtarg), batch_size=batch_size, shuffle=True, drop_last=True)
            optim = torch.optim.Adam(qfunc.parameters_last_only(), lr=lr)
            for o, a, qt in data_loader:
                optim.zero_grad()
                q = qfunc.forward_from_old_cascade_features(o).gather(dim=1, index=a)
                loss_fct(q, qt).backward()
                optim.step()

plt.figure()
plt.plot(msbes_train)
plt.legend(['MSBE'])
plt.show()
msbes_train = np.asarray(msbes_train)
np.save('msbe_pendulum.npy', msbes_train)
