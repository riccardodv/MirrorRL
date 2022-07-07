import numpy as np
import warnings
import torch


def merge_data_(d1, d2, max_len):
    if not d1:
        for key in d2.keys():
            d1[key] = d2[key]
    else:
        for key in d1.keys():
            d1[key] = torch.vstack([d1[key], d2[key]])
            d1[key] = d1[key][-max_len:]


class EnvWithTerminal:
    def __init__(self, env):
        self.env = env
        self.horizon = self.env._max_episode_steps
        self.env._max_episode_steps = np.inf
        self.done_steps = 0

    def reset(self):
        self.done_steps = 0
        return self.env.reset()

    def step(self, act):
        ret = self.env.step(act)
        terminal = done = ret[2]
        self.done_steps += 1
        if self.done_steps >= self.horizon:
            done = True
            if terminal:
                warnings.warn('reached horizon, and terminal is suspiciously True. Check if _max_episode_steps ' +
                              'has the expected behavior in your environment')
        return ret[0], ret[1], done, terminal

    def render(self):
        self.env.render()

    def get_nb_act(self):
        return self.env.action_space.n

    def get_dim_obs(self):
        return self.env.observation_space.shape[0]

    def seed(self, seed):
        self.env.seed(seed)


class Sampler:
    def __init__(self, env):
        self.curr_rollout = []
        self.policy = None
        self.env = env

    def _rollout(self, render=False, device='cpu'):
        # Generates SARSA type transitions until episode's end
        obs = self.env.reset()
        obs_tensor = torch.FloatTensor(obs, device=device)
        act = self.policy(obs_tensor)
        done = False
        while not done:
            if render:
                self.env.render()
            nobs, rwd, done, terminal = self.env.step(act)
            nobs_tensor = torch.FloatTensor(nobs, device=device)
            nact = self.policy(nobs_tensor)
            yield obs, act, rwd, done, terminal, nobs, nact
            obs = nobs
            act = nact

    def rollouts(self, policy, min_trans, max_trans, render=False, device='cpu'):
        # Keep calling rollout and saving the resulting path until at least min_trans transitions are collected
        assert (min_trans <= max_trans)
        self.policy = policy
        keys = ['obs', 'act', 'rwd', 'done', 'terminal', 'nobs', 'nact']  # must match order of the yield above
        paths = {}
        for k in keys:
            paths[k] = []
        max_reached = False
        while len(paths['rwd']) < min_trans:
            for trans_vals in self.curr_rollout:
                for key, val in zip(keys, trans_vals):
                    paths[key].append(val)
                if len(paths['rwd']) >= max_trans:
                    max_reached = True
                    break
            if not max_reached:
                self.curr_rollout = self._rollout(render, device)

        for key in set(keys):
            paths[key] = torch.FloatTensor(np.asarray(paths[key]), device=device)
            if paths[key].ndim == 1:
                paths[key] = paths[key].unsqueeze(1)
        return paths


def update_logging_stats(rwds, dones, curr_cum_rwd, returns_list, total_ts):
    for r, d in zip(rwds, dones):
        curr_cum_rwd += r[0]
        if d:
            returns_list.append(curr_cum_rwd)
            curr_cum_rwd = 0
    return curr_cum_rwd, returns_list, total_ts + rwds.shape[0]


def softmax_policy(obs, qfunc, eta, squeeze_out=True):
    with torch.no_grad():
        if obs.ndim < 2:
            obs = torch.tensor(obs)[None, :]
        if squeeze_out:
            return torch.distributions.Categorical(logits=eta * qfunc(obs)).sample().squeeze(0).cpu().numpy()
        else:
            return torch.distributions.Categorical(logits=eta * qfunc(obs)).sample().unsqueeze(1)


def get_targets_qvals(q_values_next, rwd, done, discount, lam):
    q_targets = torch.zeros_like(q_values_next)
    for k in reversed(range(len(q_values_next))):
        if done[k] or k == len(q_values_next) - 1:  # this is a new path
            q_targets[k] = rwd[k] + discount * q_values_next[k]
        else:
            q_targets[k] = (1 - lam) * (rwd[k] + discount * q_values_next[k])\
                               + lam * (rwd[k] + discount * q_targets[k + 1])
    return q_targets
