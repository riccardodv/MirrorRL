# MirrorRL

## To launch rlberry agents
 
1. Create config corresponding to environment in `configs/env`
2. Create config corresponding to agent solving this environment in the folder corresponding to an agent, e.g. `configs/a2c/solving_cartpole.yaml`
3. Add this agent and environment in `configs`, e.g. `configs/solving_cartpole.yaml` in the last line for a new agent
4. Launch experiment `python run_experiment.py configs/solving_cartpole.yaml --n_fit 1`
