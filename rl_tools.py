import numpy as np
import warnings
import torch


def merge_data_(d1, d2, max_len):
    if not d1:
        for key in d2.keys():
            d1[key] = d2[key]
    else:
        for key in d1.keys():
            d1[key] = np.append(d1[key], d2[key], axis=0)
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

    def _rollout(self, render=False):
        # Generates SARSA type transitions until episode's end
        obs = self.env.reset()
        act = self.policy(obs)
        done = False
        while not done:
            if render:
                self.env.render()
            nobs, rwd, done, terminal = self.env.step(act)
            nact = self.policy(nobs)
            yield obs, act, rwd, done, terminal, nobs, nact
            obs = nobs
            act = nact

    def rollouts(self, policy, min_trans, max_trans, render=False):
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
                self.curr_rollout = self._rollout(render)

        for key in set(keys):
            paths[key] = np.asarray(paths[key])
            if paths[key].ndim == 1:
                paths[key] = np.expand_dims(paths[key], axis=-1)
        return paths


def update_logging_stats(rwds, dones, curr_cum_rwd, returns_list, total_ts):
    for r, d in zip(rwds, dones):
        curr_cum_rwd += r[0]
        if d:
            returns_list.append(curr_cum_rwd)
            curr_cum_rwd = 0
    return curr_cum_rwd, returns_list, total_ts + rwds.shape[0]


def softmax_policy(obs, qfunc, eta):
    with torch.no_grad():
        obs = torch.tensor(obs)[None, :]
        return torch.distributions.Categorical(logits=eta * qfunc(obs)).sample().squeeze(0).numpy()


def stable_kl_div(old_probs, new_probs, epsilon=1e-12):
    new_probs = new_probs + epsilon
    old_probs = old_probs + epsilon
    kl = new_probs*torch.log(new_probs) - new_probs*torch.log(old_probs)
    return kl