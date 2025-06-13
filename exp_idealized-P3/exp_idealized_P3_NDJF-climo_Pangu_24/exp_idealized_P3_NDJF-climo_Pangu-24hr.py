#====================================================#
# experiment: Idealized MJO-P3 initilized from NDJF climatology
#====================================================#
import os
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
start = timeit.default_timer()

#===== load and set up model =====#
ort_session = load_pangu()

#===== read perturbation =====#
ds = xr.open_dataset('../Delta_T_MJO_P3_single-term.nc')
d_T = ds.d_T.to_numpy().astype(np.float32)
d_T_1K = d_T/2.5

#===== read climo tendency =====#
path_climo = '/barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/climo_runs/output/NDJF-climo/'
v_srf_0 = np.load(path_climo + 'input_srf_NDJF-climo_1980-2023_day0.npy').astype(np.float32)
v_srf_1 = np.load(path_climo + 'output_srf_NDJF-climo_1980-2023_day1.npy').astype(np.float32)
v_upper_0 = np.load(path_climo + 'input_upper_NDJF-climo_1980-2023_day0.npy').astype(np.float32)
v_upper_1 = np.load(path_climo + 'output_upper_NDJF-climo_1980-2023_day1.npy').astype(np.float32)

td_srf = v_srf_1 - v_srf_0
td_upper = v_upper_1 - v_upper_0

#===== create output data =====#
output_data_dir = '../data/exp_idealized-P3-NDJF-climo_Pangu'
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

# Load the upper-air numpy arrays
input_data_dir = '../data/ERA5/NDJF_climo'
input = np.load(os.path.join(input_data_dir, 'input_upper_NDJF-climo_1980-2023_day0.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_srf_NDJF-climo_1980-2023_day0.npy')).astype(np.float32)

print(type(input_surface))
print(type(input))


#===== Run the inference session =====#
for i in range(25):
    print('day ' + str(i)) 
    
    #===== add heating =====#
    input[2, ] = np.where(np.isnan(d_T_1K), input[2, ], input[2, ] + d_T_1K)
    if i == 0:
        np.save(os.path.join(output_data_dir, 'output_upper_NDJF-climo_MJO_P3_full-heating-1K_steady_day0'), input)

    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    
    #===== remove tendency =====#
    output = output - td_upper
    output_surface = output_surface - td_srf
    
    # Save the results
    np.save(os.path.join(output_data_dir, 'output_upper_day' + str(i+1)), output)
    np.save(os.path.join(output_data_dir, 'output_srf_day' + str(i+1)), output_surface)

    input, input_surface = output, output_surface
    del output, output_surface
   
#===== soft link initial condition to output folder =====#
input_file_srf = input_data_dir + '/input_srf_NDJF-climo_1980-2023_day0.npy'
output_file_srf = output_data_dir + '/input_srf_NDJF-climo_MJO_P3_full-heating-1K_steady_day0.npy'
input_file = os.path.abspath(input_file_srf)
output_file = os.path.abspath(output_file_srf)
os.symlink(input_file, output_file)

#===== counting time =====#
stop = timeit.default_timer()
print('running time: ', stop - start)  

#====================================================#
# postprocessing
#====================================================#
postproc_pangu.output_processing(output_data_dir, 'idealized-P3-NDJF-climo', rm_npy=True)