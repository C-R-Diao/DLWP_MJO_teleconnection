import gcsfs
import os
import jax
import numpy as np
import pickle
import xarray as xr

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

from datetime import datetime, timedelta
import time
#==========================================================#
# load model
gcs = gcsfs.GCSFileSystem(token='anon')
model_name = 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'
with gcs.open(f'gs://gresearch/neuralgcm/04_30_2024/{model_name}', 'rb') as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

#==========================================================#
# load input and forcing data
print('Loading input and forcing data...')
start_time = time.time()

fpath = '/barnes-engr-scratch1/dcr17/NeuralGCM/experiments/idealized_T/ERA5_climo/'
fname_input = 'ERA5_input_NDJF-climo_1980-2023.nc'
fname_forcing = 'ERA5_forcing_NDJF-climo_1980-2023.nc'

ds_input = xr.open_dataset(fpath + fname_input)
ds_forcing = xr.open_dataset(fpath + fname_forcing)

#===== regrid =====#
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=ds_input.sizes['latitude'],
    longitude_nodes=ds_input.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(ds_input.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(ds_input.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)
input_era5 = xarray_utils.regrid(ds_input, regridder)
input_era5 = xarray_utils.fill_nan_with_nearest(input_era5)

forcing_era5 = xarray_utils.regrid(ds_forcing, regridder)
forcing_era5 = xarray_utils.fill_nan_with_nearest(forcing_era5)

#===== deal with input data dimension to need requirement =====#
input_era5 = input_era5.sortby('level')
time_coord = np.array(['2023-12'], dtype='datetime64[ns]')
input_era5 = input_era5.expand_dims(time=time_coord)
forcing_era5 = forcing_era5.expand_dims(time=time_coord)


# print(input_era5)
# print(forcing_era5)

#===== add heating at day0 =====#
fname_dT = '/barnes-engr-scratch1/dcr17/NeuralGCM/experiments/idealized_T/P3_pert_1K_steady_NDJF-climo/Delta_T_MJO_P3_single-term.nc'
ds_dT = xr.open_dataset(fname_dT)

dT_grid = spherical_harmonic.Grid(
    latitude_nodes=ds_dT.sizes['latitude'],
    longitude_nodes=ds_dT.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(ds_dT.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(ds_dT.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    dT_grid, model.data_coords.horizontal, skipna=True
)
dT = xarray_utils.regrid(ds_dT, regridder)
dT = dT.sortby('level')
dT = dT.expand_dims(time=time_coord)
d_T = dT.d_T # heating 

input_era5['temperature'] = xr.where(d_T.isnull(), input_era5['temperature'], input_era5['temperature'] + d_T)


end_time = time.time()
elapsed_time = end_time - start_time
print(f'COMPLETE: data loaded in {elapsed_time:.2f} seconds')
print('')

#==========================================================#
# forecast
print('Initializing model...')
start_time = time.time()

#===== calcualte steps =====#
inner_steps = 24 # save model outputs every 24 hours
outer_steps = 60 # save 25 outputs
time_delta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

#===== initialize model state =====#
inputs = model.inputs_from_xarray(input_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(forcing_era5.isel(time=0))
rng_key = jax.random.key(0)  # optional for deterministic models

initial_state = model.encode(inputs, input_forcings, rng_key)

#===== use constant forcing from climo =====#
all_forcings = model.forcings_from_xarray(forcing_era5)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'COMPLETE: model initialized in {elapsed_time:.2f} seconds')
print('')

#==========================================================#
# make forecast
print('Making forecast...')
start_time = time.time()

final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=time_delta,
    start_with_input=True,
)
output = model.data_to_xarray(predictions, times=times)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'COMPLETE: forecast in {elapsed_time:.2f} seconds')

#==========================================================#
# save to netcdf
print('Save to netcdf...')

path_out = '/barnes-engr-scratch1/dcr17/NeuralGCM/experiments/idealized_T/output/output_P3_1K_day0_NDJF-climo/'
fout_name = 'output_MJO_P3_1K_day0_NDJF-climo_day0-24_1-4-deg-model.nc'

if os.path.exists(path_out + fout_name):
    os.remove(path_out + fout_name)
    print(f'Removed existing output: {fout_name}')

output.to_netcdf(path_out + fout_name)
