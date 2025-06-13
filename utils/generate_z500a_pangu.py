def generate_z500a_pangu(case_name = 'Hist-MJO-P3_Pangu'):
    '''
    function to generate geopotential height at 500 hPa, 
    and remove the annual cycle by subtracting the daily climatology from 2000-2023,
    create a new netcdf for each case, can one case-mean file
    '''
    import os
    import sys
    import glob
    import numpy as np
    import xarray as xr
    from datetime import datetime, timedelta

    #===== load dates =====#
    fname = './historical-MJO-P3_cases.txt'
    dates = []

    # Open the file and read the lines
    with open(fname, 'r') as file:
        for line in file:
            if not line.startswith("#"):
                date = line.strip()
                dates.append(date)

    #===== load daily climatology =====#
    fpath_climo = '../data/ERA5/annual_cycle_climo/' # might be wrong, check preivoue path below
    # fpath_climo = '../../Pangu_experiments/ERA5_climo/'
    fname_climo = ['*month-1_2000-2023.nc', '*month-2_2000-2023.nc', '*month-3_2000-2023.nc', '*month-11_2000-2023.nc','*month-12_2000-2023.nc']

    files_climo = []
    for f in fname_climo:
        files_climo.extend(glob.glob(fpath_climo + f))

    ds_temp = [xr.open_dataset(f) for f in files_climo]
    ds_climo = xr.concat(ds_temp, dim="time")

    year_climo = 2000 # set year to 2000
    ds_climo["time"] = [datetime.strptime(f'{year_climo}-{t}', "%Y-%m-%dT%H") for t in ds_climo["time"].values]

    # add feb 29th data to climatology, set to be the same as Feb 28th
    ds_climo_feb = ds_climo.sel(time=slice(f'{year_climo}-02-28', f'{year_climo}-02-28'))
    ds_climo_feb["time"] = [datetime.strptime(f'{year_climo}-02-29T00', "%Y-%m-%dT%H")]
    ds_climo = xr.concat([ds_climo, ds_climo_feb], dim="time")  


    #===== load experiment outputs =====#
    for date in dates:
        print(date)
        fpath_exp = f'../data/exp_Hist-MJO-P3/{case_name}_{date}/'
        fout_name = f'output_upper.{case_name}_{date}.z500_anomaly.day0-30.nc'
        fin_name = f'output_upper.{case_name}_{date}.z.day0-30.nc'
        ds = xr.open_dataset(fpath_exp + fin_name)

        # read geopotential and convert to z500
        z500 = ds['z'].sel(plev=500).drop('plev') / 9.8 # convert To geopotential height (meter)
        z500 = z500.rename('z500')
        z500.attrs['units'] = 'meter'
        z500.attrs['long_name'] = '500 hPa Geopotential Height'  

        # create datetime
        start_date = datetime.strptime(date, "%Y-%m-%d")
        time = [start_date + timedelta(days=i) for i in range(z500.shape[0])]
        time1 = xr.DataArray(time, dims='time', name='time')
        z500['time'] = time1

        # remove annual cycle by subracting the daily climatology
        time_climo = [t.replace(year=year_climo) for t in time]
        z500_climo = ds_climo['z500'].sel(time=time_climo).isel(level=0)
        z500_climo['time'] = z500['time'] 
        z500a = z500 - z500_climo
        z500a.name = 'z500a'
        z500a.attrs['units'] = 'meter'
        z500a.attrs['long_name'] = '500 hPa Geopotential Height Anomaly'
        z500a.attrs['method'] = 'subtract daily climatology from 2000-2023'

        # save to netcdf file
        ds_out = xr.Dataset({
            'z500': z500,
            'z500a': z500a
        })

        if os.path.exists(fpath_exp + fout_name):
            os.remove(fpath_exp + fout_name)
        ds_out.to_netcdf(fpath_exp + fout_name)
        print(f'File saved: {fout_name}')

def create_z500a_casemean_pangu(case_name = 'Hist-MJO-P3_Pangu'):
    '''
    generate case-mean for z500 anomalies
    '''
    import os
    import sys
    import glob
    import numpy as np
    import xarray as xr

    print(f'Generating case mean: {case_name}')
    #===== load dates =====#
    fname = './historical-MJO-P3_cases.txt'
    dates = []

    # Open the file and read the lines
    with open(fname, 'r') as file:
        for line in file:
            if not line.startswith("#"):
                date = line.strip()
                dates.append(date)

    #===== load experiment outputs =====#
    files = []
    for date in dates:
        fpath_exp = f'../data/exp_Hist-MJO-P3/{case_name}_{date}/'
        f_name = f'output_upper.{case_name}_{date}.z500_anomaly.day0-30.nc'
        files.append(fpath_exp + f_name)

    ds = [xr.open_dataset(f) for f in files]
    ds_aligned = xr.align(*ds, join='override')

    ds_out = xr.concat(ds_aligned, dim="ensemble").mean(dim="ensemble")
    ds_out.attrs.update({
        "description": 'ensemble mean of historical-MJO-P3',
        "number_of_ensemble": len(ds)
    })
    
    fout_path = f'../data/exp_Hist-MJO-P3/'
    fout_name = f'output_upper.{case_name}.z500a.case-mean.day0-30.nc'
    
    if os.path.exists(fout_path + fout_name):
        os.remove(fout_path + fout_name)
    ds_out.to_netcdf(fout_path + fout_name)
