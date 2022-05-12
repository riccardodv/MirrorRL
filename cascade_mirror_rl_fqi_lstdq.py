import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rl_tools import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy
from cascade_mirror_rl_brm import CascadeQ
from msc_tools import clone_lin_model, stable_kl_div
from bipedal_discrete import BipedalWalkerDiscrete
from pendulum_discrete import PendulumDiscrete
from lstdq_torch import lstd_q


def main():
    # env_id = 'MountainCar-v0'
    # env_id = 'CartPole-v1'
    # env_id = 'DiscretePendulum'
    env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'DiscreteBipedalWalker'
    # torch.set_num_threads(1)

    print('learning on', env_id)
    if env_id == 'DiscretePendulum':
        env = PendulumDiscrete()
    elif env_id == 'DiscreteBipedalWalker':
        env = BipedalWalkerDiscrete()
    else:
        env = EnvWithTerminal(gym.make(env_id))
    env_sampler = Sampler(env)
    nb_iter = 100
    gamma = .99
    nb_samp_per_iter = 10000
    max_replay_memory_size = nb_samp_per_iter

    nb_act = env.get_nb_act()
    dim_s = env.get_dim_obs()
    cascade_qfunc = CascadeQ(dim_s, nb_act)
    min_grad_steps_per_iter = 10000
    nb_add_neurone_per_iter = 10
    neurone_non_linearity = torch.nn.Tanh()

    batch_size = 64
    lr_model = 1e-3

    data = {}
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []
    eta = .1
    for iter in range(nb_iter):
        roll = env_sampler.rollouts(lambda x: softmax_policy(x, cascade_qfunc, eta), min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)
        with torch.no_grad():
            if data:
                data['nact'] = softmax_policy(torch.FloatTensor(data['nobs']), cascade_qfunc, eta, squeeze_out=False)

        merge_data_(data, roll, max_replay_memory_size)

        obs, act, rwd, nobs, nact, not_terminal = torch.FloatTensor(data['obs']), torch.LongTensor(data['act']), \
                              torch.FloatTensor(data['rwd']), torch.FloatTensor(data['nobs']), torch.LongTensor(data['nact']),\
                                        1 - torch.FloatTensor(data['terminal'])

        print(f'iter {iter} ntransitions {total_ts} avr_return_last_20 {np.mean(returns_list[-20:])} number of terminal states {sum(data["terminal"])}')

        # pre-computations before adding neurones to cascade
        with torch.no_grad():
            obs_feat = cascade_qfunc.get_features(obs)
            nobs_feat = cascade_qfunc.get_features(nobs)
            nobs_q_all = cascade_qfunc.get_q(nobs)
            obs_q = cascade_qfunc.get_q(obs).gather(dim=1, index=act)
            nobs_q = nobs_q_all.gather(dim=1, index=nact)
            obs_old_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(obs))
            nobs_old_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(nobs))
            nobs_v = (nobs_q_all * nobs_old_distrib.probs).sum(1, keepdim=True)
            old_out = clone_lin_model(cascade_qfunc.output)
            # q_target = rwd + gamma * nobs_q * not_terminal

        # add 'nb_add_neurone_per_iter' neurones to cascade
        cascade_qfunc.add_n_neurones(obs_feat, n=nb_add_neurone_per_iter, non_linearity=neurone_non_linearity)
        # data_loader = DataLoader(TensorDataset(obs_feat, act, obs_q, q_target), batch_size=batch_size, shuffle=True, drop_last=True)
        grad_steps = 0
        while grad_steps < min_grad_steps_per_iter:
            # train
            train_losses = []
            deltas = []
            with torch.no_grad():
                newqsp = cascade_qfunc.forward_from_old_cascade_features(nobs_feat).gather(dim=1, index=nact) # q-value for the next state and actions taken in the next states
                target = rwd + gamma * (nobs_q + newqsp) * not_terminal - obs_q

                # newqsp_all = cascade_qfunc.forward_from_old_cascade_features(nobs_feat)
                # newvsp = (newqsp_all * nobs_old_distrib.probs).sum(1, keepdim=True)
                # q_target = rwd + gamma * (newvsp + nobs_v) * not_terminal

                data_loader = DataLoader(TensorDataset(obs_feat, act, target), batch_size=batch_size, shuffle=True, drop_last=True)
            optim = torch.optim.Adam([*cascade_qfunc.cascade_neurone_list[-1].parameters(), *cascade_qfunc.output.parameters()], lr=lr_model)

            for s, a, tq in data_loader:
                optim.zero_grad()
                newqs = cascade_qfunc.forward_from_old_cascade_features(s) 
                qs = newqs.gather(dim=1, index=a)
                loss = (qs - tq).pow(2).mean()
                train_losses.append(loss.item())
                loss.backward()
                optim.step()
                grad_steps += 1
            print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)}')

        def compute_msbe(add_old):
            newqs = cascade_qfunc.forward_from_old_cascade_features(obs_feat).gather(dim=1, index=act)
            newqsp = cascade_qfunc.forward_from_old_cascade_features(nobs_feat).gather(dim=1, index=nact)
            if add_old:
                msbe = rwd + gamma * (nobs_q + newqsp) * not_terminal - obs_q - newqs
            else:
                msbe = rwd + gamma * newqsp * not_terminal - newqs
            return msbe.pow(2).mean()
        print(f'Starting LSTDQ, Mean Squared Bellman Error before LSTDQ {compute_msbe(True)}')
        new_phis = cascade_qfunc.get_features_from_old_cascade_features(obs_feat)
        new_phisp = cascade_qfunc.get_features_from_old_cascade_features(nobs_feat)
        bias, weight = lstd_q(new_phis, act, rwd, new_phisp, nact, not_terminal, gamma, nb_act, add_bias=True)
        cascade_qfunc.output.bias.data = bias
        cascade_qfunc.output.weight.data = weight
        print(f'Finished LSTDQ, Mean Squared Bellman Error after LSTDQ {compute_msbe(False)}')

        cascade_qfunc.qfunc = clone_lin_model(cascade_qfunc.output)
        cascade_qfunc.merge_with_old_weight_n_bias(old_out.weight, old_out.bias)
        with torch.no_grad():
            new_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(obs))
            # kl = torch.distributions.kl_divergence(obs_old_distrib, new_distrib).mean().item()
            kl = stable_kl_div(obs_old_distrib.probs, new_distrib.probs).mean().item()
            if kl > 1e30:
                print("KL is too large!")
                print("obs_old_distrib.probs", obs_old_distrib.probs)
                print("new_distrib", new_distrib.probs)
            normalized_entropy = new_distrib.entropy().mean().item() / np.log(nb_act)
            print(f'grad_steps {grad_steps} q_error_train last epoch {np.mean(train_losses)} kl {kl} entropy (in (0, 1)) {normalized_entropy}')


if __name__ == '__main__':
    main()
