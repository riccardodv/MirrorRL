import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.data.sampler import RandomSampler, BatchSampler

from cascade.utils import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy, get_targets_qvals
from cascade.nn import CascadeQ

from cascade.nn import FeedForward, EnsembleNN
from cascade.utils import clone_lin_model, stable_kl_div
import os
import pandas as pd
from cascade.discrete_envs import PendulumDiscrete, HopperDiscrete
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

ENV_ID = "CartPole-v1"
# ENV_ID = "Acrobot-v1"
# ENV_ID = "DiscretePendulum"
# ENV_ID = "HopperDiscrete"
# ENV_ID = "MinAtar/Freeway-v0" #try with larger eta; eta = 0.1 -> reward = 6
MAX_EPOCH = 150

default_config = {
        "env_id": ENV_ID,
        "max_epoch": MAX_EPOCH,
        "nb_samp_per_iter": 10000,
        "min_grad_steps_per_iter": 10000,
        "nb_add_neurone_per_iter": 10,
        "batch_size": 32,
        "lr_model": 1e-3,
        "max_replay_memory_size": 10**4,
        "target_update_freq": 1000,
        "replay_start_size": 0,
        "eta": 0.5,
        "gamma": 0.99,
        "seed": 0,
        "max_epoch": MAX_EPOCH,
        "print_every": 1000,
        "weight_decay": 1e-3
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
    # cascade_qfunc = CascadeQ(dim_s, nb_act)
    neurone_non_linearity = torch.nn.Tanh()

    # cascade_qfunc.to(device)

    # models = []
    qfunc = FeedForward(dim_s, nb_act, [nb_add_neurone_per_iter])
    qfunc.to(device)

    ensemble_nn = EnsembleNN([])
    ensemble_qfunc = qfunc
    #print("\n\n\n DATA initially \n\n\n", [p for p in cascade_qfunc.parameters()], "\n\n\n")
    data = {}
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []

    if replay_start_size:
        roll = env_sampler.rollouts(lambda x: softmax_policy(
            x, ensemble_qfunc, eta), min_trans=replay_start_size, max_trans=replay_start_size, device = device)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(
            roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)

        merge_data_(data, roll, max_replay_memory_size)



    for iter in range(nb_iter):

        roll = env_sampler.rollouts(lambda x: softmax_policy(
            x, ensemble_qfunc, eta), min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter, device = device)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(
            roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)

        with torch.no_grad():
            if data:
                data['nact'] = softmax_policy(data['nobs'].float().to(device), ensemble_qfunc, eta, squeeze_out=False)
                #data['nact'] = softmax_policy(torch.FloatTensor(data['nobs']).to(device), cascade_qfunc, eta, squeeze_out=False)

        merge_data_(data, roll, max_replay_memory_size)

        obs, act, rwd, nobs, nact, not_terminal = data['obs'], data['act'].long(), \
            data['rwd'], data['nobs'], data['nact'].long(),\
            1 - data['terminal'].float()

        obs_old_distrib = torch.distributions.Categorical(logits=eta * ensemble_qfunc(obs))
        obs = obs.to(device)
        act = act.to(device)
        rwd = rwd.to(device)
        nobs = nobs.to(device)
        nact = nact.to(device)
        not_terminal = not_terminal.to(device)


        print(
            f'iter {iter} ntransitions {total_ts} avr_return_last_20 {np.mean(returns_list[-20:])}')


        qfunc = FeedForward(dim_s, nb_act, [nb_add_neurone_per_iter])
        if iter > 0:
            qfunc.copy_model(ensemble_nn.models[-1])
        qfunc.to(device)

        optim = torch.optim.Adam([*qfunc.parameters()], lr=lr_model, weight_decay=weight_decay)




        # grad_steps = 0
        rs = RandomSampler(range(len(obs)), replacement = True, num_samples = min_grad_steps_per_iter * batch_size)
        bs = BatchSampler(rs, batch_size, drop_last = False)
        train_losses = []
        for grad_steps, batch_idx in enumerate(bs):
            # for grad_steps in range(min_grad_steps_per_iter):
            if grad_steps % target_update_freq == 0:
                # train_losses = []
                with torch.no_grad():
                    newqsp = qfunc(nobs).gather(
                        dim=1, index=nact)  # q-value for the next state and actions taken in the next states
                    q_target = rwd + gamma * (newqsp) * not_terminal # do we really need to update Q-target for each epoch????????????
                    dataset = TensorDataset(obs, act, q_target)
            s, a, tq = dataset[batch_idx]
            optim.zero_grad()
            newqs = qfunc(s)
            qs = newqs.gather(dim=1, index=a)
            loss = (qs - tq).pow(2).mean()
            train_losses.append(loss.item())
            loss.backward()
            optim.step()

            if (grad_steps + 1) % print_every == 0:
                print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)}')
            
 
        ensemble_nn.add_model(qfunc)
        ensemble_qfunc = ensemble_nn
        # ensemble_qfunc = lambda x: ensemble(x, models)

        with torch.no_grad():
            new_distrib = torch.distributions.Categorical(
                logits=eta * ensemble_qfunc(obs))
            kl = stable_kl_div(obs_old_distrib.probs,
                               new_distrib.probs).mean().item()

            normalized_entropy = new_distrib.entropy().mean().item() / np.log(nb_act)
            print(
                f'grad_steps {grad_steps} q_error_train last epoch {np.mean(train_losses)} kl {kl} entropy (in (0, 1)) {normalized_entropy}\n')
        with tune.checkpoint_dir(iter) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((ensemble_qfunc.state_dict(), ensemble_qfunc.state_dict()), path)

        tune.report(average_reward=np.mean(returns_list[-20:]), 
                    q_error_train= (np.mean(train_losses)), 
                    kl=kl, 
                    entropy = normalized_entropy)
    if save_model_dir is not None:
        torch.save(ensemble_qfunc, os.path.join(save_model_dir, f'cascade_qfunc_{ENV_ID}.pt'))
    print("Finished Training!")


def main(num_samples=10, max_num_epochs=10, min_epochs_per_trial=10, rf= 2., gpus_per_trial=0.):

    # config for acrobot
    config = {
        "nb_samp_per_iter": tune.grid_search([10000]),
        "min_grad_steps_per_iter": tune.grid_search([10000]),
        "nb_add_neurone_per_iter": tune.grid_search([10]),
        "batch_size": tune.grid_search([64]),
        "lr_model": tune.grid_search([1e-3]),
        "max_replay_memory_size": tune.grid_search([10000]),
        #"eta": tune.loguniform(0.1, 10),
        "eta": tune.grid_search([0.1]), # the smaller the better, best around 0.1, 0.5
        "gamma": tune.grid_search([0.99]),
        "seed": tune.grid_search([1]),
        "nb_inputs": tune.grid_search([-1]), #-1 if you want full cascade, otherwise specify nb_neurons to be connected to, including input
        "env_id" : tune.grid_search([ENV_ID]), 
        "max_epoch": tune.grid_search([MAX_EPOCH])
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

    result = tune.run(
        run,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose = 0)

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
    # main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 0.5)
    #main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 0.)
    run(default_config, save_model_dir='models')
