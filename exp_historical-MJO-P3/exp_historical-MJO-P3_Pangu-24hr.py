#====================================================#
# experiment: Historical MJO-P3 initilized from individual MJO phase 3 cases
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

#===== read cases =====#
fname = '/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/historical-MJO-P3_cases.txt'
dates = []

with open(fname, 'r') as file:
    for line in file:
        if not line.startswith("#"):
            date = line.strip()
            dates.append(date)
print(f'{len(dates)} cases in total')

#===== loop over cases =====#
# for date in dates:
for date in dates:
    print(f'Running case: {date}')
    
    #===== run simulation ======#
    path_in = '/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/ERA5_subset/initial_hist-MJO-P3_Pangu/'
    f_input_plev = path_in + f'input_upper_{date}.npy'
    f_input_srf = path_in + f'input_srf_{date}.npy'
    path_out = f'/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/outputs/Hist-MJO-P3_Pangu_{date}/'
    os.mkdir(path_out)
    
    run_pangu(ort_session, f_input_plev, f_input_srf, path_out, run_length=30)

    #===== postprocessing =====#
    postproc_pangu.output_processing(path_out, f'Hist-MJO-P3_Pangu_{date}', rm_npy=True)
    
