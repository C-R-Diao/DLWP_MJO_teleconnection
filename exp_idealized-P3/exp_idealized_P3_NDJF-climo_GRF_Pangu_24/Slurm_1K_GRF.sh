#!/bin/bash

#SBATCH --partition=bar_gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=P1KS
#SBATCH --output=pangu_GRF.txt
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=chenrui.diao@colostate.edu

pwd
source /barnes-engr-scratch1/dcr17/python_venv/venv_pangu/bin/activate
pip list
###python /barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/1K_steady_GRF/run_pangu_1K_GRF.py 
python /barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/1K_steady_GRF/run_pangu_1K_GRF_extended_20241120.py
deactivate
