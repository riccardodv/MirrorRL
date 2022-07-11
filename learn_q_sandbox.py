from re import S
import numpy as np
import torch
from rl_tools import Sampler, EnvWithTerminal
import gym
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


def nb_params_cascade(ns, k, nn, na):
    return k * nn * ns + max(0, nn * nn * k * (k - 1) / 2) + (ns + k * nn) * na


def nb_params_cascade_v2(ns, k, nn, na):
    return k * nn * ns + max(0, nn * k * (k - 1) / 2) + (ns + nn) * k + (ns + k) * na


def compute_mc_returns(rolls, gamma):
    assert rolls['done'][-1], 'trajectories have to end with done'
    mc = torch.zeros_like(rolls['rwd'])
    for k in range(len(rolls['rwd'])):
        ri = len(rolls['rwd']) - k - 1
        if rolls['done'][ri]:
            mc[ri] = rolls['rwd'][ri]
        else:
            mc[ri] = rolls['rwd'][ri] + gamma * mc[ri + 1]  # trajectories have to end with done
    return mc


def deterministic_policy(obs, cascade_q):
    with torch.no_grad():
        return torch.argmax(cascade_q(obs[None, :])).cpu().numpy()


class MLP(torch.nn.Module):
    def __init__(self, sizes, non_lin):
        super().__init__()
        ops = []
        for si, so in zip(sizes[:-1], sizes[1:]):
            ops.append(torch.nn.Linear(si, so))
            ops.append(non_lin)
        self.f = torch.nn.Sequential(*ops[:-1])

    def forward(self, x):
        return self.f(x)


env_name = ['pendulum', 'acrobot'][0]
if env_name == 'pendulum':
    pol = torch.load('models/cascade_qfunc_DiscretePendulum_iter15.pt')
    env = PendulumDiscrete(horizon=1200)
elif env_name == 'acrobot':
    pol = torch.load('models/cascade_qfunc_Acrobot-v1_iter15.pt')
    env = EnvWithTerminal(gym.make('Acrobot-v1'))

env.seed(0)
neurone_per_iter = 10
nb_iter = 100
print('nb params cascade at end of training:', nb_params_cascade(env.get_dim_obs(), nb_iter, neurone_per_iter, env.get_nb_act()))
print('nb params cascade v2 at end of training:', nb_params_cascade_v2(env.get_dim_obs(), nb_iter, neurone_per_iter, env.get_nb_act()))

env_sampler = Sampler(env)
nb_samp = 10000
gamma = .99
device = torch.device('cpu')
if env_name == 'pendulum':
    rolls_learn = env_sampler.rollouts(policy=lambda x: deterministic_policy(x, pol), min_trans=nb_samp * 4, max_trans=np.inf, device=device)
    true_q = compute_mc_returns(rolls_learn, gamma)
    nb_trans = len(true_q)
    nb_trajs = int(sum(rolls_learn['done']).item())
    rolls_new = {'rwd': [], 'terminal': [], 'obs': [], 'nobs': [], 'act': [], 'nact': []}
    true_q_new = []
    for key in rolls_new.keys():
        for t in range(nb_trajs):
            rolls_new[key].append(rolls_learn[key][1200 * t:1200 * t + 200])
        rolls_learn[key] = torch.vstack(rolls_new[key])
    for t in range(nb_trajs):
        true_q_new.append(true_q[1200 * t:1200 * t + 200])
    true_q = torch.vstack(true_q_new)

else:
    rolls_learn = env_sampler.rollouts(policy=lambda x: deterministic_policy(x, pol), min_trans=nb_samp, max_trans=np.inf, device=device)
    true_q = compute_mc_returns(rolls_learn, gamma)

non_linearity = torch.nn.Tanh()
learn_mode = ['fqi', 'lstd', 'mc', 'mc_mlp'][1]
# rolls_test = env_sampler.rollouts(policy=lambda x: deterministic_policy(x, pol),
#                              min_trans=nb_samp, max_trans=nb_samp, device=device)
returns = rwds_finished_rollouts(rolls_learn)
print(returns, len(rolls_learn['rwd']))
# plt.plot(returns, 'x')
# plt.show()

if env_name == 'acrobot':
    assert all(~rolls_learn['done'].bool() | rolls_learn['terminal'].bool()), 'has a rollout that did not reach a terminal state, aborting run'

rwd, ter, obs, nobs, act, nact = rolls_learn['rwd'], rolls_learn['terminal'].float(), rolls_learn['obs'],\
                                 rolls_learn['nobs'], rolls_learn['act'].long(), rolls_learn['nact'].long()

lr = 1e-3
batch_size = 64

if learn_mode == 'mc_mlp':
    qfunc = MLP([env.get_dim_obs(), 123, 123, env.get_nb_act()], non_linearity)
    optim = torch.optim.Adam(qfunc.parameters(), lr=lr)
    data_loader = DataLoader(TensorDataset(obs, act, true_q), batch_size=batch_size, shuffle=True, drop_last=True)
    print('nb params NN', sum([a.numel() for a in qfunc.parameters()]))
else:
    qfunc = CascadeNN(env.get_dim_obs(), env.get_nb_act())

epoch_per_iter = 10
loss_fct = torch.nn.MSELoss()
msbes_train = []
mse_train = []
for it in range(nb_iter):
    if learn_mode == "lstd":
        # computing Q target and bellman errors
        with torch.no_grad():
            old_phis = qfunc.get_features(obs)
            old_nphis = qfunc.get_features(nobs)
            old_qtarg = rwd + gamma * (1 - ter) * qfunc.output(old_nphis).gather(dim=1, index=nact)
            old_qvals = qfunc.output(old_phis).gather(dim=1, index=act)
            e_so = old_qvals - old_qtarg
            e_o = e_so.mean(dim=0).repeat(len(e_so), 1)


    if not learn_mode == 'mc_mlp':
        with torch.no_grad():
            obs_features = qfunc.get_features(obs)
            nobs_features = qfunc.get_features(nobs)
        qfunc.add_n_neurones(obs_features, torch.inf, neurone_per_iter, non_linearity=non_linearity, init_from_old=True)


    if learn_mode == 'mc':
        optim = torch.optim.Adam(qfunc.parameters_last_only(), lr=lr)
    if learn_mode == 'lstd':
        # initialize optimizer just for linear output layer
        optim = torch.optim.Adam(qfunc.feat_last_only(), lr=lr)
        # compute features when adding new neurones
  


        # for i in range(10): 
        #     v_s = qfunc.get_features(obs)
        #     v = v_s.mean(dim=0)
        #     corr = -((v_s - v)*(e_so- e_o)).sum(dim=0).abs().sum()
        #     corr.backward()
        #     optim.step()
        #     optim.zero_grad()
        
        data_loader = DataLoader(TensorDataset(obs, e_so, e_o), batch_size=batch_size, shuffle=True, drop_last=True)

        for e in range(epoch_per_iter):
            for _obs, _e_so, _e_o in data_loader:
                _v_s = qfunc.get_features(_obs) # TODO check that this is the right thing to do, shouldnt it be just features from last neurons added?
                _v = _v_s.mean(dim=0)
                corr = -((_v_s - _v)*(_e_so - _e_o)).sum(dim=0).abs().sum()
                corr.backward()
                optim.step()
                optim.zero_grad()


        phis = qfunc.get_features(obs)
        nphis = qfunc.get_features(nobs)

        bias, weight = lstd_q(phis, act, rwd, nphis, nact, 1 - ter, gamma, env.get_nb_act(), add_bias=True)
        qfunc.output.bias.data, qfunc.output.weight.data = bias, weight

        # logging
        qtarg = rwd + gamma * (1 - ter) * qfunc.output(nphis).gather(dim=1, index=nact)
        qvals = qfunc.output(phis).gather(dim=1, index=act)
        msbes_train.append(loss_fct(qvals, qtarg).item())
        mse_train.append(loss_fct(qvals, true_q).item())
        print(f'iter {it}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}')

    elif learn_mode == 'fqi':
        for e in range(epoch_per_iter):
            with torch.no_grad():
                qtarg = rolls_learn['rwd'] + gamma * (1 - ter) * qfunc.forward_from_old_cascade_features(nobs_features).gather(dim=1, index=nact)
                # logging
                qvals = qfunc.forward_from_old_cascade_features(obs_features).gather(dim=1, index=act)
                msbes_train.append(loss_fct(qvals, qtarg).item())
                mse_train.append(loss_fct(qvals, true_q).item())
                print(f'iter {it}, epoch {e}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}')

            data_loader = DataLoader(TensorDataset(obs_features, act, qtarg), batch_size=batch_size, shuffle=True, drop_last=True)
            optim = torch.optim.Adam(qfunc.parameters_last_only(), lr=lr)
            for o, a, qt in data_loader:
                optim.zero_grad()
                q = qfunc.forward_from_old_cascade_features(o).gather(dim=1, index=a)
                loss_fct(q, qt).backward()
                optim.step()
    elif learn_mode == 'mc':
        for e in range(epoch_per_iter):
            with torch.no_grad():
                qtarg = rolls_learn['rwd'] + gamma * (1 - ter) * qfunc.forward_from_old_cascade_features(nobs_features).gather(dim=1, index=nact)
                # logging
                qvals = qfunc.forward_from_old_cascade_features(obs_features).gather(dim=1, index=act)
                msbes_train.append(loss_fct(qvals, qtarg).item())
                mse_train.append(loss_fct(qvals, true_q).item())
                print(f'iter {it}, epoch {e}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}')

            data_loader = DataLoader(TensorDataset(obs_features, act, true_q), batch_size=batch_size, shuffle=True, drop_last=True)
            for o, a, qt in data_loader:
                optim.zero_grad()
                q = qfunc.forward_from_old_cascade_features(o).gather(dim=1, index=a)
                loss_fct(q, qt).backward()
                optim.step()

    elif learn_mode == 'mc_mlp':
        for e in range(epoch_per_iter):
            with torch.no_grad():
                qtarg = rolls_learn['rwd'] + gamma * (1 - ter) * qfunc(nobs).gather(dim=1, index=nact)
                # logging
                qvals = qfunc(obs).gather(dim=1, index=act)
                msbes_train.append(loss_fct(qvals, qtarg).item())
                mse_train.append(loss_fct(qvals, true_q).item())
                print(f'iter {it}, epoch {e}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}, lr {optim.param_groups[0]["lr"]}')

            for o, a, qt in data_loader:
                optim.zero_grad()
                q = qfunc(o).gather(dim=1, index=a)
                loss_fct(q, qt).backward()
                optim.step()
    if (it + 1) % 10 == 0:
        if learn_mode == 'mc_mlp':
            optim.param_groups[0]['lr'] /= 2

plt.figure()
plt.semilogy(msbes_train)
plt.semilogy(mse_train)
plt.legend(['MSBE', 'MSE to true Q'])
plt.show()
msbes_train = np.asarray(msbes_train)
np.save(f'errors_{env_name}_{learn_mode}.npy', {'msbe': msbes_train, 'mse': mse_train})
