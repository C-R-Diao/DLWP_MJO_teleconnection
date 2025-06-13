#============================================#
def output_processing(fpath, case_name, rm_npy=False):
    '''
    convert npy outputs to netcdf files
    '''
    import numpy as np
    import xarray as xr
    import glob
    import os
    
    varnames_srf = ['msl', 'u10', 'v10', 't2m']
    varnames_plv = ['z', 'q', 't', 'u', 'v']
    print(fpath)

    #===== load coordinates =====#
    d0 = xr.open_dataset('../data/Pangu_coordinates.nc')
    lat = d0.latitude
    lon = d0.longitude
    plev = d0.plev

    total_days = int(len(glob.glob(fpath + '*srf*.npy')))
    time = np.arange(total_days)

    #===== surface =====#
    print('Read surface data....')
    print(str(total_days) + ' days in total (including initial condition)')

    for varname in varnames_srf:
        print('Reading: ' + varname)

        v = np.empty((total_days, lat.shape[0], lon.shape[0]))
        v.fill(np.nan)

        fnames_srf = [fpath + f'*srf_*day_{day}.npy' for day in range(total_days)]
        #===== read npy files =====#
        for day, fname_srf in enumerate(fnames_srf):
            
            # print('loading day:' + str(day))
            fname_srf = glob.glob(fname_srf)
            ds = np.load(fname_srf[0])
            v[day, ] = ds[varnames_srf.index(varname)]

        #===== create xarray dataarray =====#
        v = xr.DataArray(v,
                         dims=['time', 'latitude', 'longitude'],
                         coords = {'time': time,
                                   'latitude': lat,
                                   'longitude': lon
                         },
                         attrs = d0[varname].attrs
        )
        
        #===== save to netcdf =====#
        v = v.to_dataset(name = varname)
        output_path = f"{fpath}output_srf.{case_name}.{varname}.day0-{total_days - 1}.nc"
        v.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
        
        del ds

    
    #===== upper =====#
    print('read upper data....')
    for varname in varnames_plv:
        print('reading ' + varname)
        print(str(total_days) + ' days in total)')

        v = np.empty((total_days, plev.shape[0], lat.shape[0], lon.shape[0]))
        v.fill(np.nan)  # Fill with NaN initially

        fanmes_paths = [fpath + f'*upper*day_{day}.npy' for day in range(total_days)]
        
        #===== read npy files =====#
        for day, fname_upper in enumerate(fanmes_paths):

            # print('loading day:' + str(day))
            files = glob.glob(fname_upper)
            ds = np.load(files[0])
            v[day, ] = ds[varnames_plv.index(varname)]

        #===== create xarray dataarray =====#
        v = xr.DataArray(v,
                         dims=['time', 'plev', 'latitude', 'longitude'],
                         coords = {'time': time,
                                   'plev': plev,
                                   'latitude': lat,
                                   'longitude': lon
                         },
                         attrs = d0[varname].attrs
        )
        
        #===== save to netcdf =====#
        v = v.to_dataset(name = varname)
        output_path = f'{fpath}output_upper.{case_name}.{varname}.day0-{total_days-1}.nc'
        v.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
        
        del ds

    #===== remove npy files (keep the 1st and last day for continue run) =====#
    if rm_npy == True:
        print('Removing npy files')

        for day_i in range(1, total_days):
            npy_files = glob.glob(fpath + '*_day_' + str(day_i) + '.npy')
            for npy_file in npy_files:
                os.remove(npy_file)


