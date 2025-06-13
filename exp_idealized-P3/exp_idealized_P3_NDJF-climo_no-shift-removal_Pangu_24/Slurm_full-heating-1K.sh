#!/bin/bash

#SBATCH --partition=bar_gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=P1KS
#SBATCH --output=pangu_climo.txt
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=chenrui.diao@colostate.edu

pwd
source /barnes-engr-scratch1/dcr17/python_venv/venv_pangu/bin/activate
pip list
python /barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/P3_pert/P3_pert_1K_steady_NDJF-climo_not-remove-tendency/run_pangu_full-heating-1K.py
deactivate
