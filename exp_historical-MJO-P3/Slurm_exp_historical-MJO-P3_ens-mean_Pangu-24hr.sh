#!/bin/bash

#SBATCH --partition=bar_gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=P3_P
#SBATCH --output=Log_pangu_historical-mjo-p3.txt
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=chenrui.diao@colostate.edu

pwd
source /barnes-engr-scratch1/dcr17/python_venv/venv_pangu/bin/activate
## pip list
python /barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/exp_historical-MJO-P3_ens-mean_Pangu-24hr.py
deactivate
