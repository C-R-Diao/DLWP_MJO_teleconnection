#====================================================#
# experiment: Historical MJO-P3 initilized from ensemble-mean
#====================================================#
import os
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np
import xarray as xr
import timeit

# Add utils directory to Python path
utils_path = os.path.abspath("../utils")
sys.path.append(utils_path)

import postproc_pangu 
from run_pangu import *



#====================================================#
# start simulation
#====================================================#
#===== load and set up model =====#
ort_session = load_pangu()

print(f'Running case: ens-mean')

#===== run simulation ======#
path_in = '/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/ERA5_subset/initial_hist-MJO-P3_Pangu/'
f_input_plev = path_in + f'input_upper_ens-mean.npy'
f_input_srf = path_in + f'input_srf_ens-mean.npy'
path_out = f'/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/outputs/Hist-MJO-P3_Pangu_ens-mean/'
os.mkdir(path_out)

run_pangu(ort_session, f_input_plev, f_input_srf, path_out, run_length=30)

#====================================================#
# postprocessing
#====================================================#
postproc_pangu.output_processing(path_out, f'Hist-MJO-P3_Pangu_ens-mean', rm_npy=True)
    
