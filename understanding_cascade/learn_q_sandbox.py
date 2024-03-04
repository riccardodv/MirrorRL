from re import S
import numpy as np
import torch
from cascade.utils import Sampler, EnvWithTerminal
import gymnasium as gym
from cascade.discrete_envs import PendulumDiscrete
import matplotlib.pyplot as plt
from cascade.nn import CascadeNN, CascadeNNBN
from torch.utils.data import DataLoader, TensorDataset
from cascade.utils import lstd_q
import os

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
    load_path = os.path.dirname(__file__)
    load_path = os.path.join(load_path, "models", "cascade_qfunc_DiscretePendulum_iter_15.pth")
    pol = torch.load(load_path)
    env = PendulumDiscrete(horizon=1200)
elif env_name == 'acrobot':
    load_path = os.path.dirname(__file__)
    load_path = os.path.join(load_path, "models", "cascade_qfunc_Acrobot-v1_iter15.pt")
    pol = torch.load(load_path)
    env = EnvWithTerminal(gym.make('Acrobot-v1'))

env.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(3)
neurone_per_iter = 10
nb_iter = 50
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

non_linearity = torch.nn.ReLU()
learn_mode = ['fqi_bn', 'fqi', 'lstd', 'mc', 'mc_mlp', "lstd_random"][0]
# rolls_test = env_sampler.rollouts(policy=lambda x: deterministic_policy(x, pol),
#                              min_trans=nb_samp, max_trans=nb_samp, device=device)

idx = torch.arange(len(rolls_learn["obs"]))
train_idx, test_idx = torch.utils.data.random_split(idx, [len(idx) - int(len(idx) * 0.1), int(len(idx) * 0.1)])


rolls_learn_test = dict()
for k in rolls_learn.keys():
    rolls_learn_test[k] = rolls_learn[k][test_idx]


for k in rolls_learn.keys():
    rolls_learn[k] = rolls_learn[k][train_idx]

true_q_test = true_q[test_idx]
true_q = true_q[train_idx]



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
elif learn_mode == 'fqi_bn':
    qfunc = CascadeNNBN(env.get_dim_obs(), env.get_nb_act())
else:
    qfunc = CascadeNN(env.get_dim_obs(), env.get_nb_act())

epoch_per_iter = 30
loss_fct = torch.nn.MSELoss()
msbes_train = []
mse_train = []

msbes_test = []
mse_test = []

for it in range(nb_iter):
    if learn_mode == "lstd":
        with torch.no_grad():
            old_phis = qfunc.get_features(obs)
            old_nphis = qfunc.get_features(nobs)

            old_qtarg = rwd + gamma * (1 - ter) * qfunc.output(old_nphis).gather(dim=1, index=nact)
            old_qvals = qfunc.output(old_phis).gather(dim=1, index=act)
            e_so = old_qvals - old_qtarg
            e_o = e_so.mean(dim=0)
            e_o_std = e_so.std(dim=0)

    if learn_mode == 'fqi_bn':
        qfunc.train(False)
        with torch.no_grad():
            obs_features = qfunc.get_features(obs)
            nobs_features = qfunc.get_features(nobs)
        qfunc.add_n_neurones(neurone_per_iter)

    elif not learn_mode == 'mc_mlp':
        # computing Q target
        with torch.no_grad():
            obs_features = qfunc.get_features(obs)
            nobs_features = qfunc.get_features(nobs)
        qfunc.add_n_neurones(obs_features, torch.inf, neurone_per_iter, non_linearity=non_linearity, init_from_old=True)


    if learn_mode == 'mc':
        optim = torch.optim.Adam(qfunc.parameters_last_only(), lr=lr)
    if learn_mode == 'lstd':
        optim = torch.optim.Adam(qfunc.feat_last_only(), lr=lr)
        # sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.99)

        for i in range(100): 

            # v_s = qfunc.get_features(obs)

            feat = qfunc.get_features(obs)
            v_s = qfunc.get_features_from_old_cascade_features( feat, stack=False)

            v = v_s.mean(dim=0)
            v_std =  v_s.std(dim=0)
            corr = -((v_s - v)*(e_so- e_o)/(v_std*e_o_std*v_s.shape[0])).sum(dim=0).abs().sum()
            if (i+1)%10 == 0:
                print("corr=", corr, "cov", ((v_s - v)*(e_so- e_o)).sum(dim=0).abs().sum())
            corr.backward()
            optim.step()
            optim.zero_grad()
            # sched.step()


        
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

    if learn_mode == 'lstd_random':
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
                # msbes_train.append(loss_fct(qvals, qtarg).item())
                # mse_train.append(loss_fct(qvals, true_q).item())
                # print(f'iter {it}, epoch {e}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}')

            data_loader = DataLoader(TensorDataset(obs_features, act, qtarg), batch_size=batch_size, shuffle=True, drop_last=True)
            optim = torch.optim.Adam(qfunc.parameters_last_only(), lr=lr)
            for o, a, qt in data_loader:
                optim.zero_grad()
                q = qfunc.forward_from_old_cascade_features(o).gather(dim=1, index=a)
                loss_fct(q, qt).backward()
                optim.step()
        
        # logging
        with torch.no_grad():
            phis = qfunc.get_features(obs)
            nphis = qfunc.get_features(nobs)
            qtarg = rwd + gamma * (1 - ter) * qfunc.output(nphis).gather(dim=1, index=nact)
            qvals = qfunc.output(phis).gather(dim=1, index=act)
            msbes_train.append(loss_fct(qvals, qtarg).item())
            mse_train.append(loss_fct(qvals, true_q).item())
            print(f'iter {it}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}')

        #testing
        with torch.no_grad():
            test_obs = rolls_learn_test["obs"]
            test_act = rolls_learn_test["act"].long()
            test_nobs = rolls_learn_test["nobs"]
            test_nact = rolls_learn_test["nact"].long()
            test_rwd = rolls_learn_test["rwd"]
            test_ter = rolls_learn_test["terminal"]

            phis = qfunc.get_features(test_obs)
            nphis = qfunc.get_features(test_nobs)
            qtarg = test_rwd + gamma * (1 - test_ter) * qfunc.output(nphis).gather(dim=1, index=test_nact)
            qvals = qfunc.output(phis).gather(dim=1, index=test_act)
            msbes_test.append(loss_fct(qvals, qtarg).item())
            mse_test.append(loss_fct(qvals, true_q_test).item())
            print(f'TEST: \t iter {it}: msbe {msbes_test[-1]:5.3f}, mse to q* {mse_test[-1]:5.3f}')

    elif learn_mode == 'fqi_bn':
        data_loader = DataLoader(TensorDataset(obs_features, act, rwd, ter, nobs_features, nact),
                                 batch_size=batch_size, shuffle=True, drop_last=True)
        optim = torch.optim.Adam(qfunc.parameters(), lr=lr)
        qfunc.train(True)
        for e in range(epoch_per_iter):
            for o, a, r, t, no, na in data_loader:
                optim.zero_grad()
                q_all = qfunc.forward_from_frozen_features(torch.cat([o, no], dim=0))
                q = q_all[:len(o), :].gather(dim=1, index=a)
                qnext = q_all[len(o):, :].gather(dim=1, index=na).detach()
                targ = r + gamma * (1 - t) * qnext
                loss_fct(q, targ).backward()
                optim.step()

        # logging
        with torch.no_grad():
            qfunc.train(False)
            qtarg = rwd + gamma * (1 - ter) * qfunc(nobs).gather(dim=1, index=nact)
            qvals = qfunc(obs).gather(dim=1, index=act)
            msbes_train.append(loss_fct(qvals, qtarg).item())
            mse_train.append(loss_fct(qvals, true_q).item())
            print(f'iter {it}: msbe {msbes_train[-1]:5.3f}, mse to q* {mse_train[-1]:5.3f}')

        # testing
        with torch.no_grad():
            qfunc.train(False)
            test_obs = rolls_learn_test["obs"]
            test_act = rolls_learn_test["act"].long()
            test_nobs = rolls_learn_test["nobs"]
            test_nact = rolls_learn_test["nact"].long()
            test_rwd = rolls_learn_test["rwd"]
            test_ter = rolls_learn_test["terminal"]

            qtarg = test_rwd + gamma * (1 - test_ter) * qfunc(test_nobs).gather(dim=1, index=test_nact)
            qvals = qfunc(test_obs).gather(dim=1, index=test_act)
            msbes_test.append(loss_fct(qvals, qtarg).item())
            mse_test.append(loss_fct(qvals, true_q_test).item())
            print(f'TEST: \t iter {it}: msbe {msbes_test[-1]:5.3f}, mse to q* {mse_test[-1]:5.3f}')

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

np.save(f"test_errors_{env_name}_{learn_mode}.npy", {'msbe': msbes_test, 'mse': mse_test})
