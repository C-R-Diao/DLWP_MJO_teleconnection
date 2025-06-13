def load_pangu(device='gpu', model_path='/barnes-engr-scratch1/dcr17/Pangu_experiments/models/pangu_weather_24.onnx'):
    '''
    initialize pangu model;
    use gpu by default
    '''
    import os
    import torch
    import onnx
    import onnxruntime as ort
    import numpy as np
    import xarray as xr
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 4

    # Set the behavier of cuda provider
    if device == 'gpu':
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}
        providers = [('CUDAExecutionProvider', cuda_provider_options)]
    else:
        providers = []

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    return ort_session

def run_pangu(ort_session, f_input_plev, f_input_srf, path_out, run_length=30):
    import os
    import numpy as np
    import xarray as xr

    #===== load initial condition =====#
    input_plev = np.load(f_input_plev).astype(np.float32)
    input_srf = np.load(f_input_srf).astype(np.float32)

    #====== auto-regressive =====#
    for i in range(run_length):
        output, output_surface = ort_session.run(None, {'input':input_plev, 'input_surface':input_srf})
        np.save(f'{path_out}output_upper_day_{i+1}', output)
        np.save(f'{path_out}output_srf_day_{i+1}', output_surface)     
        input_plev, input_srf = output, output_surface
        del output, output_surface

    #===== soft link input data to output dir =====#
    output_file_srf = f'{path_out}output_srf_day_0.npy'
    os.symlink(f_input_srf, output_file_srf)
    output_file = f'{path_out}output_upper_day_0.npy'
    os.symlink(f_input_plev, output_file)