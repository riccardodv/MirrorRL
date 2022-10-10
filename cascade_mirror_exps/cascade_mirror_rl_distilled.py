import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.data.sampler import RandomSampler, BatchSampler

from cascade.utils import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy, get_targets_qvals
from cascade.nn import CascadeQ, Simple_Cascade
from cascade.utils import clone_lin_model, stable_kl_div
import os
import pandas as pd
from cascade.discrete_envs import PendulumDiscrete, HopperDiscrete
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ENV_ID = "CartPole-v1"
# ENV_ID = "Acrobot-v1"
# ENV_ID = "DiscretePendulum"
# ENV_ID = "HopperDiscrete"
ENV_ID = "MinAtar/Freeway-v0" #try with larger eta; eta = 0.1 -> reward = 6
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
        "when_to_distill": np.arange(20, MAX_EPOCH, 20),
        "distill_epochs": 100,
    }
 


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
    when_to_distill = config["when_to_distill"]
    distill_epochs = config["distill_epochs"]
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
    cascade_qfunc = CascadeQ(dim_s, nb_act)
    neurone_non_linearity = torch.nn.Tanh()

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

        # if iter in when_to_distill:
        #     print("____DISTILLING____")
        #     new_cascade = Simple_Cascade(cascade_qfunc.dim_input, cascade_qfunc.dim_output, [cascade_qfunc.nb_hidden])
        #     new_cascade.merge_with_old_weight_n_bias(model.output.weight, model.output.bias)
        #     optim_distill = torch.optim.Adam([*new_cascade.cascade.parameters()], lr=1e-3)
        #     #train new cascade using the dataset and other cascade

        #     # should I initialize the weights from the last output? For now yes
        #     # the compression of weights should be done more in relative based knowledge distillation
        #     data_loader_train = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True, drop_last=True)
        #     # old_w, olb_b = new_cascade.output.weight, new_cascade.output.bias
        #     for i in range(distill_epochs):
        #         # print("mse between old w and cur w: ", loss(old_w, new_cascade.output.weight))
        #         for x, y in data_loader_train:
        #             new_cascade_features = new_cascade.get_features(x)
        #             old_cascade_features = model.get_features(x)
        #             loss_feat = loss(new_cascade_features, old_cascade_features)
        #             loss_out = loss(new_cascade(x), y)
        #             final_loss = loss_feat + loss_out
        #             final_loss.backward()
        #             optim_distill.step()
            
        #     #distilled model is trained
        #     model = new_cascade

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


        print(
            f'iter {iter} ntransitions {total_ts} avr_return_last_20 {np.mean(returns_list[-20:])}')

        # add new neurone to cascade
        with torch.no_grad():
            obs_feat = cascade_qfunc.get_features(obs)
            nobs_feat = cascade_qfunc.get_features(nobs)
            print("Device of nobs", nobs.device)
            nobs_q_all = cascade_qfunc.get_q(nobs)
            obs_q = cascade_qfunc.get_q(obs).gather(dim=1, index=act)
            nobs_q = nobs_q_all.gather(dim=1, index=nact)
            obs_old_distrib = torch.distributions.Categorical(
                logits=eta * cascade_qfunc(obs))
            nobs_old_distrib = torch.distributions.Categorical(
                logits=eta * cascade_qfunc(nobs))
            nobs_v = (nobs_q_all * nobs_old_distrib.probs).sum(1, keepdim=True)
            old_out = clone_lin_model(cascade_qfunc.output)
            # q_target = rwd + gamma * nobs_q * not_terminal

        # cascade_qfunc.add_n_neurones(obs_feat, nb_inputs=nb_add_neurone_per_iter + dim_s,
        #                              n_neurones=nb_add_neurone_per_iter, non_linearity=neurone_non_linearity)
        nbInputs = obs_feat.shape[1]
        if "nb_inputs" in config.keys():
            if config["nb_inputs"] >= dim_s:
                nbInputs = config["nb_inputs"]

        cascade_qfunc.add_n_neurones(obs_feat, nb_inputs=nbInputs, n_neurones=nb_add_neurone_per_iter, non_linearity=neurone_non_linearity)
        # cascade_qfunc.add_n_neurones(obs_feat, n=nb_add_neurone_per_iter)
        cascade_qfunc.to(device)


        optim = torch.optim.Adam([*cascade_qfunc.cascade_neurone_list[-1].parameters(),
                                      *cascade_qfunc.output.parameters()], lr=lr_model)

        # grad_steps = 0
        rs = RandomSampler(range(len(obs_feat)), replacement = True, num_samples = min_grad_steps_per_iter * batch_size)
        bs = BatchSampler(rs, batch_size, drop_last = False)
        train_losses = []
        for grad_steps, batch_idx in enumerate(bs):
            # for grad_steps in range(min_grad_steps_per_iter):
            if grad_steps % target_update_freq == 0:
                # train_losses = []
                with torch.no_grad():
                    newqsp = cascade_qfunc.forward_from_old_cascade_features(nobs_feat).gather(
                        dim=1, index=nact)  # q-value for the next state and actions taken in the next states
                    q_target = rwd + gamma * (nobs_q + newqsp) * not_terminal - obs_q 
                    dataset = TensorDataset(obs_feat, act, q_target)
            s, a, tq = dataset[batch_idx]
            optim.zero_grad()
            newqs = cascade_qfunc.forward_from_old_cascade_features(s)
            qs = newqs.gather(dim=1, index=a)
            loss = (qs - tq).pow(2).mean()
            train_losses.append(loss.item())
            loss.backward()
            optim.step()

            if (grad_steps + 1) % print_every == 0:
                print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)}')
            
 

        cascade_qfunc.merge_q(old_out)
        with torch.no_grad():
            new_distrib = torch.distributions.Categorical(
                logits=eta * cascade_qfunc(obs))
            kl = stable_kl_div(obs_old_distrib.probs,
                               new_distrib.probs).mean().item()

            normalized_entropy = new_distrib.entropy().mean().item() / np.log(nb_act)
            print(
                f'grad_steps {grad_steps} q_error_train last epoch {np.mean(train_losses)} kl {kl} entropy (in (0, 1)) {normalized_entropy}\n')
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
