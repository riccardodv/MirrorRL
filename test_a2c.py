import gym
import rlberry
import numpy as np
import torch
from rlberry.agents.torch import A2CAgent
from rlberry.envs import gym_make
from rlberry.manager import plot_writer_data, AgentManager
from rl_tools import EnvWithTerminal
from rlberry.agents.torch.utils.training import *


try_envs = ["MountainCar-v0", "CartPole-v1", "Acrobot-v1", "LunarLander-v2"]




def main():

    ########## A2C agent ###########    

    # env_id = 'MountainCar-v0'
    # env_id = 'CartPole-v1'
    env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    env_kwargs = dict(id = env_id)

    torch.set_num_threads(1)
    # env = EnvWithTerminal(gym.make(env_id))
    policy_configs = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (256,),
        "reshape": False,
        "is_policy": True,
    }

    critic_configs = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (512,),
        "reshape": False,
        "out_size": 1,
    }


    def model_factory_from_size_configs(env, **kwargs):
        """
        Returns a default Q value network.
        """
        # model_config = size_model_config(env, **kwargs)
        return model_factory(size_model_config(env, **kwargs))

    A2Cagent = AgentManager(A2CAgent, 
            (gym_make, env_kwargs),
            fit_budget=1000,
            init_kwargs=dict(policy_net_fn=mlp, 
                            policy_net_kwargs=policy_configs, 
                            value_net_fn = mlp, 
                            value_net_kwargs=critic_configs,
                            learning_rate=7e-4,
                            optimizer_type = "RMS_PROP",
                            horizon = 500,
                            entr_coef = 0.01),
            eval_kwargs=dict(eval_horizon=200),
            n_fit=4,
            agent_name = "A2C rlberry agent"
        )
    A2Cagent.fit()

    plot_writer_data(A2Cagent, tag="episode_rewards", title="A2C. Rewards", savefig_fname="A2C reward per step")
    plot_writer_data(A2Cagent, tag="episode_rewards", title="A2C. Rewards", xtag="dw_time_elapsed", savefig_fname="A2C reward per time")




if __name__ == '__main__':
    main()
