import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cascade.utils import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy, get_targets_qvals, uniform_random_policy
from cascade.nn import CascadeQ, CascadeQ2
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
ENV_ID = "MinAtar/Breakout-v0" #try with larger eta; eta = 0.1 -> reward = 6
# ENV_ID = "MinAtar/Freeway-v0" #try with larger eta; eta = 0.1 -> reward = 6

MAX_EPOCH = 150

default_config = {
        "env_id": ENV_ID,
        "max_epoch": MAX_EPOCH,
        "nb_samp_per_iter": 10000,
        "min_grad_steps_per_iter": 10000,
        "nb_add_neurone_per_iter": 50,
        "batch_size": 64,
        "lr_model": 1e-3,
        "max_replay_memory_size": 10**4,
        "eta": 1,
        "gamma": 0.99,
        "seed": 0,
        "nb_inputs": 50 # number of inputs to connect to without the input state dimension
        # "env_id": ENV_ID,
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([2, 4, 8, 16])
    }
 



# def run(env_id='CartPole-v1',
#          nb_iter=100,
#          nb_samp_per_iter=10000,
#          min_grad_steps_per_iter=20000,
#          nb_add_neurone_per_iter=10,
#          batch_size=64,
#          lr_model=1e-3,
#          max_replay_memory_size=10000,
#          eta=.1,
#          gamma = .99
#          ):

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
    lam = 0.
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
    cascade_qfunc = CascadeQ2(dim_s, nb_act, init_nb_hidden = 0)
    neurone_non_linearity = torch.nn.Tanh()

    cascade_qfunc.to(device)

    #print("\n\n\n DATA initially \n\n\n", [p for p in cascade_qfunc.parameters()], "\n\n\n")
    data = {}
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []
    alpha = torch.nn.Parameter(torch.tensor(1.0))

    for iter in range(nb_iter):

        #TODO propose an alternative strategy for the first iter
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


        print(f'iter {iter} ntransitions {total_ts} avr_return_last_20 {np.mean(returns_list[-20:])}')

        # add new neurone to cascade
        with torch.no_grad():
            obs_feat = cascade_qfunc.get_features(obs)
            nobs_feat = cascade_qfunc.get_features(nobs)
            print("Device of nobs", nobs.device)
            if iter == 0:
                obs_q = torch.zeros((obs.shape[0], 1))
                nobs_q = torch.zeros((nobs.shape[0], 1))
                obs_old_distrib = torch.distributions.Categorical(torch.ones((obs.shape[0], 1))*1/nb_act)
            else:
                obs_q = cascade_qfunc.get_q(obs).gather(dim=1, index=act)
                nobs_q = cascade_qfunc.get_q(nobs).gather(dim=1, index=nact)
                obs_old_distrib = torch.distributions.Categorical(logits=eta * cascade_qfunc(obs))
            old_out = clone_lin_model(cascade_qfunc.output)

        # specify connectivity to previous hidden neurons
        nbInputs = obs_feat.shape[1]
        if "nb_inputs" in config.keys():
            if config["nb_inputs"] >= 0:
                nbInputs = config["nb_inputs"]
                nbInputs += dim_s

        cascade_qfunc.add_n_neurones(obs_feat, nb_inputs=nbInputs, n_neurones=nb_add_neurone_per_iter, non_linearity=neurone_non_linearity)
        cascade_qfunc.to(device)
        grad_steps = 0
        while grad_steps < min_grad_steps_per_iter:
            # train
            train_losses = []
            projected_reward_losses = []
            projected_future_losses = []
            with torch.no_grad():
                qsp = cascade_qfunc.forward_from_old_cascade_features(nobs_feat).gather(dim=1, index=nact)  # q-value for the next state and actions taken in the next states
                q_target = rwd + gamma * (nobs_q + qsp) * not_terminal - obs_q

                data_loader = DataLoader(TensorDataset(
                    obs_feat, act, q_target, obs, nobs, nact, rwd), batch_size=batch_size, shuffle=True, drop_last=True)

            optim = torch.optim.Adam([*cascade_qfunc.cascade_neurone_list[-1].parameters(),
                                      *cascade_qfunc.output.parameters()], lr=lr_model)
            for s, a, tq, o, no, na, r in data_loader:
                optim.zero_grad()
                qs = cascade_qfunc.forward_from_old_cascade_features(s).gather(dim=1, index=a)
                loss = (qs - tq).pow(2).mean()
                train_losses.append(loss.item())
                loss.backward()
                optim.step()
                # with torch.no_grad():
                #     of = cascade_qfunc.get_features(o)
                #     nof = cascade_qfunc.get_features(no)
                #     pinv = torch.linalg.pinv(of)
                #     pr_phi = of @ pinv
                #     projected_reward_losses.append((r- pr_phi @ r).pow(2).mean().item())
                #     projected_future_losses.append((nof - pr_phi @ nof).pow(2).mean().item())
                
                grad_steps += 1
                if grad_steps >= min_grad_steps_per_iter:
                    break
            print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)}')
            # print(f"\t projected rewards {np.mean(projected_reward_losses)}, projected future features {np.mean(projected_future_losses)}")






        optim_alpha = torch.optim.Adam([alpha], lr = 0.01)

        for e in range(500):
            optim_alpha.zero_grad()
            qsp= cascade_qfunc.forward_from_old_cascade_features(nobs_feat).gather(dim=1, index=nact)  # q-value for the next state and actions taken in the next states
            q_target = rwd + gamma * (nobs_q + alpha * qsp.detach()) * not_terminal - obs_q
            qs = cascade_qfunc.forward_from_old_cascade_features(obs_feat).gather(dim=1, index=act)
            loss = (alpha * qs.detach() - q_target).pow(2).mean()
            loss.backward()
            optim_alpha.step()
            print("\t \t alpha error = {}, alpha = {}".format(loss.item(), alpha.data))


        if iter == 0:
            cascade_qfunc.merge_q(old_out, alpha=1)
        else:
            cascade_qfunc.merge_q(old_out, alpha=alpha.data)

        # comment/uncomment above to switch on/off the alpha optim
        cascade_qfunc.merge_q(old_out, alpha=1)


        # weights_q = cascade_qfunc.qfunc.weight.data
        # bias_q = cascade_qfunc.qfunc.bias.data
        #########################################
        with torch.no_grad():
            of = cascade_qfunc.get_features(obs)
            # nof = cascade_qfunc.get_features(nobs)
            nq = cascade_qfunc.get_q(nobs)
            pinv = torch.linalg.pinv(of)
            pr_phi = of @ pinv
            projected_reward_losses.append((rwd- pr_phi @ rwd).pow(2).mean().item())
            projected_future_losses.append((nq  - pr_phi @ nq).pow(2).mean().item())
            
        print(f"\t projected rewards {np.mean(projected_reward_losses)}, projected future features {np.mean(projected_future_losses)}")
        #########################################


        with torch.no_grad():
            new_distrib = torch.distributions.Categorical(
                logits=eta * cascade_qfunc(obs))
            # kl = torch.distributions.kl_divergence(obs_old_distrib, new_distrib).mean().item()
            kl = stable_kl_div(obs_old_distrib.probs,
                               new_distrib.probs).mean().item()
            # if kl > 1e30:
            #     print("KL is too large!")
            #     print("obs_old_distrib.probs", obs_old_distrib.probs)
            #     print("new_distrib", new_distrib.probs)
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
    # config for cartpole
    # config = {
    #     "nb_samp_per_iter": tune.grid_search([10000]),
    #     "min_grad_steps_per_iter": tune.grid_search([10000]),
    #     "nb_add_neurone_per_iter": tune.grid_search([10]),
    #     "batch_size": tune.grid_search([64]),
    #     "lr_model": tune.grid_search([1e-3]),
    #     "max_replay_memory_size": tune.grid_search([10000]),
    #     #"eta": tune.loguniform(0.1, 10),
    #     "eta": tune.grid_search([0.1]),
    #     "gamma": tune.grid_search([0.99]),
    #     "seed": tune.grid_search([1, 11, 100, 1001, 2999])
    #     # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     # "lr": tune.loguniform(1e-4, 1e-1),
    #     # "batch_size": tune.choice([2, 4, 8, 16])
    # }
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


    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == '__main__':
    # main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 0.5)
    #main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 0.)
    run(default_config, save_model_dir='models')
