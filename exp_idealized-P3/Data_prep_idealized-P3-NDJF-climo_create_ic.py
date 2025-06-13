#====================================================#
# create initial condition files for idealized-P3 experiment
# use NDJF climatology for 1980-2023 from ERA5
#====================================================#
import xarray as xr
import numpy as np
import os

#====================================================#
# create surface npy file
#====================================================#
fpath = '../data/ERA5/NDJF_climo/'
fname_srf = 'ERA5_srf_NDJF_1980-2023.nc'
ds_srf = xr.open_dataset(fpath + fname_srf)

vname_srf = [
    'msl',
    'u10',
    'v10',
    't2m'
]

#===== create a new array for data =====#
dim_0 = len(vname_srf)
v_srf = np.empty((dim_0, ds_srf.latitude.shape[0], ds_srf.longitude.shape[0]))

count = 0
for vname in vname_srf:
    temp = ds_srf[vname].mean(dim='date')
    print(temp)
    v_srf[count,] = temp.to_numpy()
    count += 1
    del temp

#===== save to npy files =====#
np.save(fpath + 'input_srf_NDJF-climo_1980-2023_day0.npy', v_srf)
del v_srf, ds_srf

#=====================================#
# create upper npy file
#=====================================#
fname_upper = 'ERA5_upper_NDJF_1980-2023.nc'
ds_upper = xr.open_dataset(fpath + fname_upper)

vname_upper = ['z', 'q', 't', 'u', 'v']

#===== create a new array for data =====#
dim_0 = len(vname_upper)
v_upper = np.empty((dim_0, ds_upper.pressure_level.shape[0], ds_upper.latitude.shape[0], ds_upper.longitude.shape[0]))

count = 0
for vname in vname_upper:
    temp = ds_upper[vname].mean(dim='date')
    print(temp)
    v_upper[count,] = temp.to_numpy()
    count += 1
    del temp

#===== save to npy files =====#
np.save(fpath + 'input_upper_NDJF-climo_1980-2023_day0.npy', v_upper)


