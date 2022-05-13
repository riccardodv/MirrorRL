import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rl_tools import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy
from cascade_mirror_rl_brm import CascadeQ
from msc_tools import clone_lin_model, norm_squared_lin, stable_kl_div


def main():
    # env_id = 'MountainCar-v0'
    env_id = 'CartPole-v1'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    torch.set_num_threads(1)

    print('learning on', env_id)
    env = EnvWithTerminal(gym.make(env_id))
    env_sampler = Sampler(env)
    gamma = .99

    nb_act = env.get_nb_act()
    dim_s = env.get_dim_obs()
    cascade_qfunc = CascadeQ(dim_s, nb_act)
    nb_iter = 100
    nb_samp_per_iter = 10000
    min_grad_steps_per_iter = 10000
    min_td0_steps_per_iter = 0
    nb_add_neurone_per_iter = 10
    norm_weight = 0.
    neurone_nonlinearity = torch.nn.Tanh()

    batch_size = 64
    lr_model = 1e-3
    lr_td0 = 1e-3

    data = {}
    max_replay_memory_size = nb_samp_per_iter
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []
    eta = .1
    for iter in range(nb_iter):

        roll = env_sampler.rollouts(lambda x: softmax_policy(x, cascade_qfunc, eta), min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)
        merge_data_(data, roll, max_replay_memory_size)

        obs, act, rwd, nobs, nact, not_terminal = torch.FloatTensor(data['obs']), torch.LongTensor(data['act']), \
                              torch.FloatTensor(data['rwd']), torch.FloatTensor(data['nobs']), torch.LongTensor(data['nact']),\
                                        1 - torch.FloatTensor(data['terminal'])

        print(f'iter {iter} ntransitions {total_ts} avr_return_last_20 {np.mean(returns_list[-20:])}')

        # add new neurone to cascade
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

        cascade_qfunc.add_n_neurones(obs_feat, n=nb_add_neurone_per_iter, non_linearity=neurone_nonlinearity)
        # data_loader = DataLoader(TensorDataset(obs_feat, act, obs_q, q_target), batch_size=batch_size, shuffle=True, drop_last=True)
        grad_steps = 0
        # FQI
        while grad_steps < min_grad_steps_per_iter:
            # train
            train_losses = []
            with torch.no_grad():
                newqsp = cascade_qfunc.forward_from_old_cascade_features(nobs_feat).gather(dim=1, index=nact) # q-value for the next state and actions taken in the next states
                q_target = rwd + gamma * (nobs_q + newqsp) * not_terminal

                # newqsp_all = cascade_qfunc.forward_from_old_cascade_features(nobs_feat)
                # newvsp = (newqsp_all * nobs_old_distrib.probs).sum(1, keepdim=True)
                # q_target = rwd + gamma * (newvsp + nobs_v) * not_terminal

                data_loader = DataLoader(TensorDataset(obs_feat, act, obs_q, q_target), batch_size=batch_size, shuffle=True, drop_last=True)

            optim = torch.optim.Adam([*cascade_qfunc.cascade_neurone_list[-1].parameters(), *cascade_qfunc.output.parameters()], lr=lr_model)
            for s, a, oldq, tq in data_loader:
                optim.zero_grad()
                newqs = cascade_qfunc.forward_from_old_cascade_features(s)
                qs = newqs.gather(dim=1, index=a) + oldq
                loss = (qs - tq).pow(2).mean() + norm_weight * (norm_squared_lin(cascade_qfunc.output) + norm_squared_lin(cascade_qfunc.cascade_neurone_list[-1].f[0]))
                train_losses.append(loss.item())
                loss.backward()
                optim.step()
                grad_steps += 1
            print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)} norm_squared_output '
                  f'{cascade_qfunc.output.weight.pow(2).sum() + cascade_qfunc.output.bias.pow(2).sum()}')

        # TD(0)
        grad_steps_td = 0
        optim_td0 = torch.optim.SGD(cascade_qfunc.output.parameters(), lr=lr_td0)
        while grad_steps_td < min_td0_steps_per_iter:
            # train
            train_losses = []
            with torch.no_grad():
                new_obs_feat = cascade_qfunc.get_features(obs)
                new_nobs_feat = cascade_qfunc.get_features(nobs)
                data_loader = DataLoader(TensorDataset(new_obs_feat, act, rwd, new_nobs_feat, nact, not_terminal, obs_q, nobs_q), batch_size=batch_size, shuffle=True, drop_last=True)

            for phis, a, r, phisp, ap, nt, oldqs, oldqsp in data_loader:
                optim_td0.zero_grad()
                qs = cascade_qfunc.output(phis).gather(dim=1, index=a) + oldqs
                qsp = cascade_qfunc.output(phisp).gather(dim=1, index=ap) + oldqsp
                td = (r + gamma * qsp * nt).detach() - qs
                loss = td.pow(2).mean() + norm_weight * (norm_squared_lin(cascade_qfunc.output) + norm_squared_lin(cascade_qfunc.cascade_neurone_list[-1].f[0]))
                train_losses.append(loss.item())

                loss.backward()
                optim_td0.step()

                grad_steps_td += 1
            print(f'\t grad_steps_td {grad_steps_td} q_error_train {np.mean(train_losses)} norm_squared_output '
                  f'{cascade_qfunc.output.weight.pow(2).sum() + cascade_qfunc.output.bias.pow(2).sum()}')


        cascade_qfunc.merge_q(old_out)
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
