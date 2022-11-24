import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cascade.utils import EnvWithTerminal, Sampler, merge_data_, update_logging_stats, softmax_policy
from cascade.nn import FeedForward, EnsembleNN
from cascade.utils import clone_lin_model, stable_kl_div
import os
from cascade.discrete_envs import PendulumDiscrete, HopperDiscrete
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ENV_ID = "CartPole-v1"
# ENV_ID = "Acrobot-v1"
# ENV_ID = "DiscretePendulum"
# ENV_ID = "HopperDiscrete"
ENV_ID = "MinAtar/Breakout-v0" #try with larger eta; eta = 0.1 -> reward = 6
MAX_EPOCH = 150

def ensemble(x, models):
    s = 0
    for m in models:
        s = s + m(x)
    return s



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
        # "env_id": ENV_ID,
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([2, 4, 8, 16])
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
    # models = []
    qfunc = FeedForward(dim_s, nb_act, [nb_add_neurone_per_iter])
    qfunc.to(device)

    ensemble_nn = EnsembleNN([])

    # neurone_non_linearity = torch.nn.Tanh()


    ensemble_qfunc = qfunc


    data = {}
    total_ts = 0
    curr_cum_rwd = 0
    returns_list = []
    for iter in range(nb_iter):
        roll = env_sampler.rollouts(lambda x: softmax_policy(
            x, ensemble_qfunc, eta), min_trans=nb_samp_per_iter, max_trans=nb_samp_per_iter, device = device)
        curr_cum_rwd, returns_list, total_ts = update_logging_stats(
            roll['rwd'], roll['done'], curr_cum_rwd, returns_list, total_ts)

        with torch.no_grad():
            if data:
                data['nact'] = softmax_policy(data['nobs'].float().to(device), ensemble_qfunc, eta, squeeze_out=False)

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


        print(f'iter {iter} ntransitions {total_ts} avr_return_last_20 {np.mean(returns_list[-20:])}')



        qfunc = FeedForward(dim_s, nb_act, [nb_add_neurone_per_iter])
        qfunc.to(device)


        grad_steps = 0
        while grad_steps < min_grad_steps_per_iter:
            # train
            train_losses = []
            with torch.no_grad():
                qsp = qfunc(obs).gather(dim=1, index = nact)
                q_target = rwd + gamma*(qsp)*not_terminal
                # q_target = rwd + gamma * (nobs_q + qsp) * not_terminal - obs_q

                data_loader = DataLoader(TensorDataset(
                    obs, act, q_target), batch_size=batch_size, shuffle=True, drop_last=True)

            optim = torch.optim.Adam([*qfunc.parameters()], lr=lr_model)
            for s, a, tq in data_loader:
                optim.zero_grad()
                qs = qfunc(s).gather(dim=1, index=a)
                loss = (qs - tq).pow(2).mean()
                train_losses.append(loss.item())
                loss.backward()
                optim.step()
                grad_steps += 1
                if grad_steps >= min_grad_steps_per_iter:
                    break
            print(f'\t grad_steps {grad_steps} q_error_train {np.mean(train_losses)}')

        # getting other nn in the ensemble
        # models.append(qfunc)
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
    # main(1, MAX_EPOCH, MAX_EPOCH, 1.1, 0.)
    run(default_config, save_model_dir='models')
