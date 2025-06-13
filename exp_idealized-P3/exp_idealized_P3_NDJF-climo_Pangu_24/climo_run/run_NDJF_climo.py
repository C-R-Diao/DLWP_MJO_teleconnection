import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import timeit


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
# providers=['CPUExecutionProvider']


# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession('/barnes-engr-scratch1/dcr17/Pangu_experiments/models/pangu_weather_24.onnx', sess_options=options, providers=providers)


#====================================================#
start = timeit.default_timer()

#===== create output data =====#
output_data_dir = './output/NDJF-climo/' 
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

#===== read initial condition =====#
input_data_dir = '/barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/ERA5/'
fname_input_upper = 'input_upper_NDJF-climo_1980-2023_day0.npy'
input = np.load(input_data_dir + fname_input_upper ).astype(np.float32)

fname_input_srf = 'input_srf_NDJF-climo_1980-2023_day0.npy'
input_surface = np.load(input_data_dir + fname_input_srf ).astype(np.float32)

print(input.shape)
print(input_surface.shape)

#===== Run the inference session =====#

for i in range(60):
    print('day ' + str(i)) 
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    # Save the results
    np.save(os.path.join(output_data_dir, 'output_upper_NDJF-climo_1980-2023_day' + str(i+1)), output)
    np.save(os.path.join(output_data_dir, 'output_srf_NDJF-climo_1980-2023_day' + str(i+1)), output_surface)

    input, input_surface = output, output_surface
    del output, output_surface
   
#===== soft link initial condition to output folder =====#
input_file_upper = input_data_dir + fname_input_upper
output_file_upper = output_data_dir + fname_input_upper
input_file = os.path.abspath(input_file_upper)
output_file = os.path.abspath(output_file_upper)
os.symlink(input_file, output_file)

input_file_srf = input_data_dir + fname_input_srf
output_file_srf = output_data_dir + fname_input_srf
input_file = os.path.abspath(input_file_srf)
output_file = os.path.abspath(output_file_srf)
os.symlink(input_file, output_file)

#===== counting time =====#
stop = timeit.default_timer()
print('running time: ', stop - start)  
