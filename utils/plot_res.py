import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # add polygon
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

#=========================================#
def add_cyclic_point(data, coord=None, axis=-1):

    # had issues with cartopy finding utils so copied for myself
    
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = np.ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = np.ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value   

#=========================================#
def drawOnGlobe(ax, map_proj, data, lats, lons, unit, cmap='coolwarm', vmin=None, vmax=None, cbarBool=True, fastBool=False, extent='both'):

    data_crs = ccrs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons) #fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons
    
    # ADD COASTLINES
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor='None',
        edgecolor = 'k',
        linewidth=.5,
    )
    ax.add_feature(land_feature)
        
    if(fastBool):
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, shading='auto', rasterized=True)  # rasterized=True is used to avoid white boundaries
    else:
        image = ax.pcolor(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap,shading='auto')
    
    # use contourf instead
    # image = ax.contourf(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, levels=100)
    
    
    if(cbarBool):
        cb = plt.colorbar(image, shrink=.6, orientation="horizontal", pad=.02, extend=extent)
        cb.ax.tick_params(labelsize=10) 
        cb.ax.set_xlabel (unit)
    else:
        cb = None

    image.set_clim(vmin,vmax)
    
    return cb, image   