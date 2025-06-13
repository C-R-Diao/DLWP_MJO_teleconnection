import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import xarray as xr
import timeit


# The directory of your input and output data
model_24 = onnx.load('/barnes-engr-scratch1/dcr17/Pangu_experiments/models/pangu_weather_24.onnx')

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 4

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

providers = [('CUDAExecutionProvider', cuda_provider_options)]

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession('/barnes-engr-scratch1/dcr17/Pangu_experiments/models/pangu_weather_24.onnx', sess_options=options, providers=providers)

#====================================================#
# read perturbation 

ds = xr.open_dataset('../Delta_T_MJO_P3_single-term.nc')
d_T = ds.d_T.to_numpy().astype(np.float32)
d_T_1K = d_T/2.5

#====================================================#
        
start = timeit.default_timer()

#===== read climo tendency =====#
# path_climo = '/barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/climo_runs/output/NDJF-climo/'
# v_srf_0 = np.load(path_climo + 'input_srf_NDJF-climo_1980-2023_day0.npy').astype(np.float32)
# v_srf_1 = np.load(path_climo + 'output_srf_NDJF-climo_1980-2023_day1.npy').astype(np.float32)
# v_upper_0 = np.load(path_climo + 'input_upper_NDJF-climo_1980-2023_day0.npy').astype(np.float32)
# v_upper_1 = np.load(path_climo + 'output_upper_NDJF-climo_1980-2023_day1.npy').astype(np.float32)
# 
# td_srf = v_srf_1 - v_srf_0
# td_upper = v_upper_1 - v_upper_0

#===== create output data =====#
output_data_dir = './output/'
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

# Load the upper-air numpy arrays
input_data_dir = '/barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/ERA5'
input = np.load(os.path.join(input_data_dir, 'input_upper_NDJF-climo_1980-2023_day0.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_srf_NDJF-climo_1980-2023_day0.npy')).astype(np.float32)

print(type(input_surface))
print(type(input))


#===== Run the inference session =====#
for i in range(30):
    print('day ' + str(i)) 
    
    #===== add heating =====#
    input[2, ] = np.where(np.isnan(d_T_1K), input[2, ], input[2, ] + d_T_1K)
    if i == 0:
        np.save(os.path.join(output_data_dir, 'output_upper_NDJF-climo_MJO_P3_full-heating-1K_steady_day0'), input)

    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    
    #===== remove tendency =====#
#    output = output - td_upper
#    output_surface = output_surface - td_srf
    
    # Save the results
    np.save(os.path.join(output_data_dir, 'output_upper_NDJF-climo_MJO_P3_full-heating-1K_steady_day' + str(i+1)), output)
    np.save(os.path.join(output_data_dir, 'output_srf_NDJF-climo_MJO_P3_full-heating-1K_steady_day' + str(i+1)), output_surface)

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
