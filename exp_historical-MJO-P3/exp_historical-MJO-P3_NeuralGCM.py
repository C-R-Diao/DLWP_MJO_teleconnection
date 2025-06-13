import gcsfs
import os
import jax
import numpy as np
import pickle
import xarray

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

gcs = gcsfs.GCSFileSystem(token='anon')

#====================================================#
# load model 
#====================================================#
model_name = 'v1/deterministic_1_4_deg.pkl' # recommended by the authors: https://neuralgcm.readthedocs.io/en/latest/checkpoints.html
with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

#====================================================#
# read cases
#====================================================#
fname = '/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/historical-MJO-P3_cases.txt'
dates = []

with open(fname, 'r') as file:
    for line in file:
        if not line.startswith("#"):
            date = line.strip()
            dates.append(date)
print(f'{len(dates)} cases in total')

#====================================================#
# run simulations
#====================================================#

#===== load era5 =====#
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

#===== lop over cases =====#
for date in dates:
    print(f'Running case: {date}')
    date0 = date + 'T00'

    #===== create initial condition =====#
    era5 = (
        full_era5[model.input_variables + model.forcing_variables]
        .pipe(
            xarray_utils.selective_temporal_shift,
            variables=model.forcing_variables,
            # time_shift='24 hours', 
        )
        .sel(time = slice(date0, date0), drop=False)
        .compute()
    )

    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes['latitude'],
        longitude_nodes=full_era5.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    input_era5 = xarray_utils.regrid(era5, regridder)
    input_era5 = xarray_utils.fill_nan_with_nearest(input_era5)

    #===== forcast for 30 days ======#
    inner_steps = 24  # save model outputs once every 24 hours
    outer_steps = 30  # total of 25 days
    timedelta = np.timedelta64(1, 'h') * inner_steps
    times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

    #===== initialize model state =====#
    inputs = model.inputs_from_xarray(input_era5.isel(time=0))
    input_forcings = model.forcings_from_xarray(input_era5.isel(time=0))
    rng_key = jax.random.key(42)  # optional for deterministic models
    initial_state = model.encode(inputs, input_forcings, rng_key)   
    
    #===== use persistence for forcing variables (SST and sea ice cover) =====#
    all_forcings = model.forcings_from_xarray(input_era5)  

    #===== make forecast =====#
    print('start forecasting')
    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )         

    #===== model output =====#
    print('forecasting done')
    output = model.data_to_xarray(predictions, times=times)
    output.attrs['initial_date'] = date0

    path_out = f'/barnes-engr-scratch1/dcr17/PROJ_DLWP_and_MJO-teleconnection/exp_historical-MJO-P3/outputs/Hist-MJO-P3_NeuralGCM_{date}_test/'
    fname_out = f'output.Hist-MJO-P3_NeuralGCM_{date}.test-no-shift.day0-30.nc'
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    output.to_netcdf(path_out + fname_out)