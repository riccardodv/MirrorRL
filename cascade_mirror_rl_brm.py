import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cascade_nn import CascadeNN
from rl_tools import EnvWithTerminal, Sampler, merge_data_


def clone_lin_model(m):
    ret = nn.Linear(m.in_features, m.out_features)
    ret.weight.data = m.weight.detach().clone()
    ret.bias.data = m.bias.detach().clone()
    return ret


class CascadeQ(CascadeNN):
    def __init__(self, dim_input, dim_output):
        super().__init__(dim_input, dim_output, init_nb_hidden=0)
        self.qfunc = clone_lin_model(self.output)

    def get_q(self, obs):
        return self.qfunc(self.get_features(obs))

    def merge_q(self, old_output_model):
        self.merge_with_old_weight_n_bias(self.qfunc.weight, self.qfunc.bias)
        self.qfunc = clone_lin_model(self.output)

        self.output.weight.data[:, :old_output_model.weight.shape[1]] += old_output_model.weight
        self.output.bias.data += old_output_model.bias


def update_logging_stats(rwds, dones, curr_cum_rwd, returns_list, total_ts):
    for r, d in zip(rwds, dones):
        curr_cum_rwd += r[0]
        if d:
            returns_list.append(curr_cum_rwd)
            curr_cum_rwd = 0
    return curr_cum_rwd, returns_list, total_ts + rwds.shape[0]


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
    nb_iter = 200
    nb_samp_per_iter = 3000
    min_grad_steps_per_iter = 10000
    nb_add_neurone_per_iter = 10

    batch_size = 64
    lr_model = 1e-3

    data = {}
    max_replay_memory_size = 64000
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []
    eta = 1.
    for iter in range(nb_iter):

        def softmax_policy(obs):
            with torch.no_grad():
                obs = torch.tensor(obs)[None, :]
                return torch.distributions.Categorical(logits=eta * cascade_qfunc(obs)).sample().squeeze(0).numpy()
        roll = env_sampler.rollouts(softmax_policy, min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter)
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
            obs_q = cascade_qfunc.get_q(obs).gather(dim=1, index=act)
            obs_old_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(obs))
            nobs_old_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(nobs))
            nobs_v = (cascade_qfunc.get_q(nobs) * nobs_old_distrib.probs).sum(1, keepdim=True)
            old_out = clone_lin_model(cascade_qfunc.output)

        cascade_qfunc.add_n_neurones(obs_feat, n=nb_add_neurone_per_iter)
        optim = torch.optim.Adam([*cascade_qfunc.cascade_neurone_list[-1].parameters(), *cascade_qfunc.output.parameters()], lr=lr_model)
        data_loader = DataLoader(
            TensorDataset(obs_feat, act, rwd, nobs_feat, nact, obs_q, nobs_v, nobs_old_distrib.probs, not_terminal),
            batch_size=batch_size, shuffle=True, drop_last=True)
        grad_steps = 0
        while grad_steps < min_grad_steps_per_iter:
            # train
            train_losses = []
            for s, a, r, sp, ap, oldq, oldvp, oldprobp, n_ter in data_loader:
                optim.zero_grad()
                newqs = cascade_qfunc.forward_from_old_cascade_features(s)
                newqsp = cascade_qfunc.forward_from_old_cascade_features(sp)
                qs = newqs.gather(dim=1, index=a) + oldq
                vsp = (newqsp * oldprobp).sum(1, keepdim=True) + oldvp
                target = r + gamma * vsp * n_ter
                loss = (qs - target).pow(2).mean()
                train_losses.append(loss.item())
                loss.backward()
                optim.step()
                grad_steps += 1
            print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)}')

        cascade_qfunc.merge_q(old_out)
        with torch.no_grad():
            new_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(obs))
            kl = torch.distributions.kl_divergence(obs_old_distrib, new_distrib).mean().item()
            normalized_entropy = new_distrib.entropy().mean().item() / np.log(nb_act)
            print(f'grad_steps {grad_steps} q_error_train last epoch {np.mean(train_losses)} kl {kl} entropy (in (0, 1)) {normalized_entropy}')


if __name__ == '__main__':
    main()
