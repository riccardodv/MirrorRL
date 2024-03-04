#! /bin/bash

module purge
module load conda/23.5.0

conda deactivate
conda activate micarl

module load cuda/11.4.0_gcc-10.4.0   
module load gcc/10.4.0_gcc-10.4.0  
module load cudnn/8.2.4.15-11.4_gcc-10.4.0


cd $HOME/MirroRL/cascade_mirror_exps/

python cascade_mirror_rl_fqi_bn.py $1

# oarsub -l host=1,walltime=1:45 "pinns_acrobot.py --out $HOME/results"