#!/bin/bash

#SBATCH --partition=bar_gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=NGCM_14
#SBATCH --output=Log_neuralgcm.log
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=chenrui.diao@colostate.edu

pwd
source /home/dcr17/miniconda/etc/profile.d/conda.sh
conda init
conda activate venv_neuralgcm
python /barnes-engr-scratch1/dcr17/NeuralGCM/experiments/idealized_T/P3_pert_1K_day0_NDJF-climo/run_neuralgcm_MJO-P3_1k_day0_NDJF-climo.py
conda deactivate
