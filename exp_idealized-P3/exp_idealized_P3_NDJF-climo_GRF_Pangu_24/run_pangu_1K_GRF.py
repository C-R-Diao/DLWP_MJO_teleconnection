import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import xarray as xr
import timeit
from utils.GRF_forcing_pangu import *


# The directory of your input and output data
input_data_dir = '/barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/ERA5'
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
# months = ['11', '12', '01', '02']
# days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
hr = 'T00'
month = '11'
day = '01'

# for month in months[0:1]:
#     for day in days[0:1]:

#====================================================# 
# create green's function for subtropical
start = timeit.default_timer()

lats = [0, 15, 30]
lons = np.arange(0, 360, 40)

for lat_c in lats:
    for lon_c in lons:
        print("heating centered at: " + str(lat_c) + ', ' + str(lon_c))

        #===== read perturbation =====#
        A = 1.0
        heating = GRF_forcing_pangu(A, lat_c, lon_c)
        d_T = heating.to_numpy().astype(np.float32)

        #===== create output data =====#
        output_data_dir = './output/1K_steady_lat-' + str(lat_c) + '_lon-' + str(lon_c)
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        
        #===== save heating map =====#
        plot_GRF_forcing_pangu(heating, output_data_dir + '/GRF_forcing_lat-' + str(lat_c) + '_lon-' + str(lon_c))

        #===== read climo tendency =====#
        path_climo = '/barnes-engr-scratch1/dcr17/Pangu_experiments/idealized_T_ens/climo_runs/output/' + month + day + '/'
        v_srf_0 = np.load(path_climo + 'input_srf_' + month + '-' + day + '_climo_1980-2023_day0.npy')  
        v_srf_1 = np.load(path_climo + 'output_srf_' + month + '-' + day + '_climo_1980-2023_day1.npy')  
        v_upper_0 = np.load(path_climo + 'input_upper_' + month + '-' + day + '_climo_1980-2023_day0.npy')  
        v_upper_1 = np.load(path_climo + 'output_upper_' + month + '-' + day + '_climo_1980-2023_day1.npy')  

        td_srf = v_srf_1 - v_srf_0
        td_upper = v_upper_1 - v_upper_0

        # Load the upper-air numpy arrays
        input = np.load(os.path.join(input_data_dir, 'input_upper_' + month + '-' + day + '_climo_1980-2023_day0.npy')).astype(np.float32)
        # Load the surface numpy arrays
        input_surface = np.load(os.path.join(input_data_dir, 'input_srf_' + month + '-' + day + '_climo_1980-2023_day0.npy')).astype(np.float32)

        #===== Run the inference session =====#
        for i in range(25):
            print('day ' + str(i)) 
            
            #===== add heating =====#
            input[2, ] = np.where(np.isnan(d_T), input[2, ], input[2, ] + d_T)
            if i == 0:
                np.save(os.path.join(output_data_dir, 'output_upper_' + month + '-' + day + '_1K_GRF_lat-' + str(lat_c) + '_lon-' + str(lon_c) + '_day0'), input)

            output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
            
            #===== remove tendency =====#
            output = output - td_upper
            output_surface = output_surface - td_srf
            
            # Save the results
            np.save(os.path.join(output_data_dir, 'output_upper_' + month + '-' + day + '_1K_GRF_lat-' + str(lat_c) + '_lon-' + str(lon_c) + '_day' + str(i+1)), output)
            np.save(os.path.join(output_data_dir, 'output_srf_' + month + '-' + day + '_1K_GRF_lat-' + str(lat_c) + '_lon-' + str(lon_c) + '_day' + str(i+1)), output_surface)

            input, input_surface = output, output_surface
            del output, output_surface
            
        #===== soft link initial condition to output folder =====#
        input_file_srf = input_data_dir + '/input_srf_' + month + '-' + day + '_climo_1980-2023_day0.npy'
        output_file_srf = output_data_dir + '/input_srf_' + month + '-' + day + '_1K_GRF_lat-' + str(lat_c) + '_lon-' + str(lon_c) + '_day0.npy'
        input_file = os.path.abspath(input_file_srf)
        output_file = os.path.abspath(output_file_srf)
        os.symlink(input_file, output_file)

        #===== counting time =====#
        stop = timeit.default_timer()
        print('running time: ', stop - start)  
