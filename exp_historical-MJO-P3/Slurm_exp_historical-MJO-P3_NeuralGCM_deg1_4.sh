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
python /barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/exp_historical-MJO-P3_NeuralGCM.py
conda deactivate
