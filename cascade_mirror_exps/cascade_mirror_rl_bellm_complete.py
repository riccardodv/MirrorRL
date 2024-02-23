import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import pdb

from cascade.utils import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy, get_targets_qvals, uniform_random_policy
from cascade.nn import CascadeQ, CascadeQ2
from cascade.utils import clone_lin_model, stable_kl_div
import os
import pandas as pd
from cascade.discrete_envs import PendulumDiscrete, HopperDiscrete

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ENV_ID = "CartPole-v1"
# ENV_ID = "Acrobot-v1"
# ENV_ID = "DiscretePendulum"
# ENV_ID = "HopperDiscrete"
ENV_ID = "MinAtar/Breakout-v0" #try with larger eta; eta = 0.1 -> reward = 6
MAX_EPOCH = 150

default_config = {
        "env_id": ENV_ID,
        "max_epoch": MAX_EPOCH,
        "nb_samp_per_iter": 10000,
        "min_grad_steps_per_iter": 10000,
        "nb_add_neurone_per_iter": 10,
        "batch_size": 64,
        "lr_model":5e-3,
        "max_replay_memory_size": 10**4,
        "target_update_freq": 10,
        "replay_start_size": 500,
        "eta": 0.5,
        "gamma": 0.99,
        "seed": 0,
        "max_epoch": MAX_EPOCH,
        "print_every": 100,
        "weight_decay": 1e-3,
        "lamda": 0.1
    }
 



def run(config, checkpoint_dir=None, save_model_dir=None):

    env_id = config["env_id"]
    nb_iter = config["max_epoch"]
    nb_samp_per_iter = config["nb_samp_per_iter"]
    min_grad_steps_per_iter = config["min_grad_steps_per_iter"]
    nb_add_neurone_per_iter = config["nb_add_neurone_per_iter"]
    batch_size = config["batch_size"]
    lr_model = config["lr_model"]
    max_replay_memory_size = config["max_replay_memory_size"]
    eta = config["eta"]
    gamma = config["gamma"]
    replay_start_size = config["replay_start_size"]
    target_update_freq = config["target_update_freq"]
    print_every = config["print_every"]
    weight_decay = config["weight_decay"]
    lamda = config["lamda"]

    if "seed" in config.keys():
        seed = config["seed"]
    else:
        seed = None

    device = "cpu"
    if torch.cuda.is_available():
        print("I can run on CUDA")
        device = "cuda:0"
   
    # env = gym.make(env_id)


    print('learning on', env_id)

    if env_id == 'DiscretePendulum':
        env = PendulumDiscrete()
    elif env_id == "HopperDiscrete":
        env = HopperDiscrete()
    else:
        env = EnvWithTerminal(gym.make(env_id))
    
    if seed is not None:
        torch.manual_seed(seed)
        env.env.seed(seed)
    
    env_sampler = Sampler(env)
    

    nb_act = env.get_nb_act()
    dim_s = env.get_dim_obs()
    cascade_qfunc = CascadeQ2(dim_s, nb_act, init_nb_hidden =nb_add_neurone_per_iter)
    # neurone_non_linearity = torch.nn.SiLU()
    neurone_non_linearity = torch.nn.ReLU()
    # neurone_non_linearity = torch.nn.Tanh()

    cascade_qfunc.to(device)

    #print("\n\n\n DATA initially \n\n\n", [p for p in cascade_qfunc.parameters()], "\n\n\n")
    data = {}
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []

    if replay_start_size:
        roll = env_sampler.rollouts(lambda x: softmax_policy(
            x, cascade_qfunc, eta), min_trans=replay_start_size, max_trans=replay_start_size, device = device)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(
            roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)

        merge_data_(data, roll, max_replay_memory_size)



    for iter in range(nb_iter):

        if iter == 0:
            roll = env_sampler.rollouts(lambda x: uniform_random_policy(
                x, nb_act), min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter, device = device)
        else:        
            roll = env_sampler.rollouts(lambda x: softmax_policy(
                x, cascade_qfunc, eta), min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter, device = device)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(
            roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)

        with torch.no_grad():
            if data:
                data['nact'] = softmax_policy(data['nobs'].float().to(device), cascade_qfunc, eta, squeeze_out=False)
                #data['nact'] = softmax_policy(torch.FloatTensor(data['nobs']).to(device), cascade_qfunc, eta, squeeze_out=False)

        merge_data_(data, roll, max_replay_memory_size)

        obs, act, rwd, nobs, nact, not_terminal = data['obs'], data['act'].long(), \
            data['rwd'], data['nobs'], data['nact'].long(),\
            1 - data['terminal'].float()

        obs = obs.to(device)
        act = act.to(device)
        rwd = rwd.to(device)
        nobs = nobs.to(device)
        nact = nact.to(device)
        not_terminal = not_terminal.to(device)


        print(f'iter {iter} ntransitions {total_ts} avr_return_last_100 {np.mean(returns_list[-100:])}')

        # add new neurone to cascade
        with torch.no_grad():
            obs_feat_data = cascade_qfunc.get_features(obs)
            nobs_feat_data = cascade_qfunc.get_features(nobs)
            print("Device of nobs", nobs.device)
            if iter == 0:
                obs_q = torch.zeros((obs.shape[0], 1)).to(device)
                nobs_q = torch.zeros((nobs.shape[0], 1)).to(device)
                obs_old_distrib = torch.distributions.Categorical(torch.ones((obs.shape[0], 1))*1/nb_act)
            else:
                obs_q = cascade_qfunc.get_q(obs).gather(dim=1, index=act).to(device)
                nobs_q = cascade_qfunc.get_q(nobs).gather(dim=1, index=nact).to(device)
                obs_old_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(obs))
            old_out = clone_lin_model(cascade_qfunc.output)


        # specify connectivity to previous hidden neurons
        nbInputs = obs_feat_data.shape[1]
        if "nb_inputs" in config.keys():
            if config["nb_inputs"] >= dim_s:
                nbInputs = config["nb_inputs"]

        cascade_qfunc.add_n_neurones(obs_feat_data, nb_inputs=nbInputs, n_neurones=nb_add_neurone_per_iter, non_linearity=neurone_non_linearity)
        cascade_qfunc.to(device)

        optim = torch.optim.Adam([*cascade_qfunc.cascade_neurone_list[-1].parameters(),
                                      *cascade_qfunc.output.parameters()], lr=lr_model, weight_decay=weight_decay)
        
        rs = RandomSampler(range(len(obs_feat_data)), replacement = True, num_samples = min_grad_steps_per_iter * batch_size)
        bs = BatchSampler(rs, batch_size, drop_last = False)
        train_losses = []
        train_regularizer = []
        for grad_steps, batch_idx in enumerate(bs):
            if grad_steps % target_update_freq == 0:
                with torch.no_grad():
                    nobs_delta = cascade_qfunc.forward_from_old_cascade_features(nobs_feat_data).gather(dim=1, index=nact) 
                    delta_target_dataset = rwd + gamma * (nobs_q + nobs_delta) * not_terminal - obs_q 
                    dataset = TensorDataset(obs_feat_data, act, delta_target_dataset, rwd, nobs_feat_data, nact, not_terminal, obs_q, nobs_q)
                    # print(f'\t \t Target nn is updated')


            obs_feat, a, delta_target, r, nobs_feat, ap, nt, q, nq  = dataset[batch_idx]
            optim.zero_grad()

            obs_feat_delta = cascade_qfunc.get_features_from_old_cascade_features_cropped(obs_feat)
            nobs_feat_delta = cascade_qfunc.get_features_from_old_cascade_features_cropped(nobs_feat) 

            delta_sa = cascade_qfunc.output(obs_feat_delta)
            delta_s = delta_sa.gather(dim=1, index=a)
            
            # cascade_qfunc.output.weight.requires_grad = False
            # cascade_qfunc.output.bias.requires_grad = False
            delta_sa_old = cascade_qfunc.forward_from_features_without_grad(nobs_feat_delta)
            delta_s_old = delta_sa_old.gather(dim=1, index=ap)
            # cascade_qfunc.output.weight.requires_grad = True
            # cascade_qfunc.output.bias.requires_grad = True

            loss1 = (delta_s - delta_target).pow(2).mean()
            loss2 = (delta_s - (r + gamma * (nq + delta_s_old) *nt - q )).pow(2).mean() 


            train_losses.append(loss1.item())
            train_regularizer.append(loss2.item())
            (loss1+ lamda * loss2).backward()
            optim.step()

            if (grad_steps + 1) % print_every == 0:
                print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)} regularizer_error {np.mean(train_regularizer)}')
            
 

        if iter == 0:
            cascade_qfunc.merge_q(old_out, alpha=1)
        else:
            cascade_qfunc.merge_q(old_out, alpha=1.) # boosting option


        # monitor the projected reward and future features
        projected_reward_losses = []
        projected_future_losses = []
        with torch.no_grad():
            of = cascade_qfunc.get_features(obs)
            # nof = cascade_qfunc.get_features(nobs)
            nq = cascade_qfunc.get_q(nobs)
            pinv = torch.linalg.pinv(of)
            pr_phi = of @ pinv
            projected_reward_losses.append((rwd- pr_phi @ rwd).pow(2).mean().item())
            projected_future_losses.append((nq  - pr_phi @ nq).pow(2).mean().item())    
            print(f"iter {iter} ntransitions {total_ts} projected rewards {np.mean(projected_reward_losses)} projected future features {np.mean(projected_future_losses)}")
        # monitor the KL divergence and entropy
            new_distrib = torch.distributions.Categorical(
                logits=eta * cascade_qfunc(obs))
            kl = stable_kl_div(obs_old_distrib.probs.to(device),
                               new_distrib.probs).mean().item()
            normalized_entropy = new_distrib.entropy().mean().item() / np.log(nb_act)
            print(f"iter {iter} ntransitions {total_ts}  grad_steps {grad_steps} q_error_train last epoch {np.mean(train_losses)} kl {kl} entropy (in (0, 1)) {normalized_entropy}\n")
        


        if ray.tune.is_session_enabled():
            with tune.checkpoint_dir(iter) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((cascade_qfunc.state_dict(), cascade_qfunc.state_dict()), path)
            tune.report(average_reward=np.mean(returns_list[-20:]), 
                        q_error_train= (np.mean(train_losses)), 
                        kl=kl, 
                        entropy = normalized_entropy)
        
    if save_model_dir is not None:
        torch.save(cascade_qfunc, os.path.join(save_model_dir, f'cascade_qfunc_{ENV_ID}.pt'))
    print("Finished Training!")


def main(num_samples=10, max_num_epochs=10, min_epochs_per_trial=10, rf= 2., gpus_per_trial=0.):

    # config for acrobot
    config = {
        "env_id" : tune.grid_search([ENV_ID]),
        "max_epoch": tune.grid_search([MAX_EPOCH]),
        "nb_samp_per_iter": tune.grid_search([10000]),
        "min_grad_steps_per_iter": tune.grid_search([10000]),
        "nb_add_neurone_per_iter": tune.grid_search([10]),
        "batch_size": tune.grid_search([64]),
        "lr_model": tune.grid_search([1e-3]),
        "max_replay_memory_size": tune.grid_search([int(1e4)]),
        "target_update_freq": tune.grid_search([100,500]),
        "replay_start_size": tune.grid_search([500]),
        "eta": tune.grid_search([0.5]), # the smaller the better, best around 0.1, 0.5
        "gamma": tune.grid_search([0.99]),
        "seed": tune.grid_search([1,2,3,4,5]),
        "print_every": tune.grid_search([100]),
        "weight_decay": tune.grid_search([1e-3]),
        "lamda": tune.grid_search([0.1, 0.5, 1., 0.05])
        # "nb_inputs": tune.grid_search([-1]), #-1 if you want full cascade, otherwise specify nb_neurons to be connected to, including input
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([2, 4, 8, 16])
    }

    scheduler = ASHAScheduler(
        metric="average_reward",
        mode="max",
        max_t=max_num_epochs,
        grace_period=min_epochs_per_trial,
        reduction_factor= rf)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["average_reward", "q_error_train", "kl", "entropy"])

    # ray.init()
    result = tune.run(
        run,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose = 0)
    # ray.shutdown()

    best_trial = result.get_best_trial("average_reward", "max", "last-10-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final q-error loss: {}".format(
        best_trial.last_result["q_error_train"]))
    print("Best trial final kl: {}".format(
        best_trial.last_result["kl"]))
    print("Best trial final entropy: {}".format(
        best_trial.last_result["entropy"]))

    print("best checkpoint dir: {}".format(
        best_trial.checkpoint.value
    ))




if __name__ == '__main__':
    main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 1.)
    # main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 0.)
    # run(default_config, save_model_dir='models') # for debugging
