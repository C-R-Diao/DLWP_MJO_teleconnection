#====================================================#
# create initial condition files for idealized-P3 experiment
# use NDJF climatology for 1980-2023 from ERA5
#====================================================#
import xarray as xr
import numpy as np
import pandas as pd
import os

# %%
#====================================================#
# load monthly surface data
#====================================================#
fpath = '../../data/ERA5/NDJF_climo/'
fname_srf = 'ERA5_srf_NDJF_1980-2023.nc'
ds_srf = xr.open_dataset(fpath + fname_srf)

#====================================================#
# load nino3.4 Index from the HadISST1.1 (url:https://psl.noaa.gov/data/timeseries/month/DS/Nino34/) 
#====================================================#
fname_nino = '../../data/ERA5/nino34.long.anom.nc'
ds_nino = xr.open_dataset(fname_nino)

nino34 = ds_nino.value.sel(time=slice('1980-01-01', '2023-12-01'))
ndjf_months = [1, 2, 11, 12]
nino34 = nino34.sel(time=nino34['time'].dt.month.isin(ndjf_months))

nino34['time'] = nino34['time'].dt.strftime('%Y%m%d').astype(int)

#===== pick el nino and la nina events =====#
el_nino = nino34.where(nino34 > 0.5,  drop=True)
la_nina = nino34.where(nino34 < -0.5, drop=True)

#====================================================#
# create surface data for enso phases
#====================================================#
ds_el_nino = ds_srf.sel(date=el_nino.time)
ds_la_nina = ds_srf.sel(date=la_nina.time)

# #===== save as netcdf files =====#
# ds_el_nino.to_netcdf(fpath + 'ERA5_srf_NDJF_1980-2023_el_nino.nc')
# ds_la_nina.to_netcdf(fpath + 'ERA5_srf_NDJF_1980-2023_la_nina.nc')

#====================================================#
# create and save surface npy file
#====================================================#
def create_npy_srf(ds_srf):
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
        print(vname)
        temp = ds_srf[vname].mean('time')
        v_srf[count,] = temp.to_numpy()
        count += 1
        del temp

    return v_srf


#===== save to npy files =====#
v_srf_el_nino = create_npy_srf(ds_el_nino)
np.save(fpath + 'input_srf_NDJF-climo_1980-2023_el-nino.npy', v_srf_el_nino)

v_srf_la_nina = create_npy_srf(ds_la_nina)
np.save(fpath + 'input_srf_NDJF-climo_1980-2023_la-nina.npy', v_srf_la_nina)

#========================================================================================================#
# load monthly plev data 
#====================================================#
fname_upper = 'ERA5_upper_NDJF_1980-2023.nc'
ds_upper = xr.open_dataset(fpath + fname_upper)

ds_upper_el_nino = ds_upper.sel(date=el_nino.time)
ds_upper_la_nina = ds_upper.sel(date=la_nina.time)

#====================================================#
# create and save upper npy file
#====================================================#
def create_npy_upper(ds_upper):
    vname_upper = ['z', 'q', 't', 'u', 'v']

    #===== create a new array for data =====#
    dim_0 = len(vname_upper)
    v_upper = np.empty((dim_0, ds_upper.pressure_level.shape[0], ds_upper.latitude.shape[0], ds_upper.longitude.shape[0]))

    count = 0
    for vname in vname_upper:
        print(vname)
        temp = ds_upper[vname].mean('time')
        v_upper[count,] = temp.to_numpy()
        count += 1
        del temp

    return v_upper

#===== save to npy files =====#
v_upper_el_nino = create_npy_upper(ds_upper_el_nino)
np.save(fpath + 'input_upper_NDJF-climo_1980-2023_el-nino.npy', v_upper_el_nino)

v_upper_la_nina = create_npy_upper(ds_upper_la_nina)
np.save(fpath + 'input_upper_NDJF-climo_1980-2023_la-nina.npy', v_upper_la_nina)



