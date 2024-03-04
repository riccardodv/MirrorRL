#! /bin/bash

YAML_FOLDER=$HOME/MirrorRL/cascade_mirror_exps/train_yamls/breakout


for f in $(ls $YAML_FOLDER/*.yaml); do
    oarsub -q production -p "cpucore > 15 and cluster in ("grappe", "grvingt", "grue")" -l host=1,walltime=16 "$HOME/MirrorRL/launch_env.sh $f"
done


# oarsub -q production -p  "cpucore > 15 and cluster in ("grappe", "grvingt", "grue")" -l host=1,walltime=16 "$HOME/continuous-rl-and-pinns/code/launch_env.sh $YAML_FOLDER/adaptive.yaml"

# oarsub -q production -p  "cpucore > 15 and cluster in ("grappe", "grvingt", "grue")" -l host=1,walltime=16 "$HOME/continuous-rl-and-pinns/code/launch_env.sh $YAML_FOLDER/adaptivea.yaml"

# oarsub -q production -p  "cpucore > 15 and cluster in ("grappe", "grvingt", "grue")" -l host=1,walltime=16 "$HOME/continuous-rl-and-pinns/code/launch_env.sh $YAML_FOLDER/adaptive_sampling.yaml"

