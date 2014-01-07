# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:17:40 2014

@author: jc
"""

import matplotlib.pyplot as plt
import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as np
from datetime import datetime,timedelta

scale = 0.03
isub = 2
tidx = -1  # to input in which day you want to forecast(third day is -1, second day is -2,first day is -3)
url = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd'

def shrink(a,b):
    """Return array shrunk to fit a specified shape by triming or averaging.
    
    a = shrink(array, shape)
    
    array is an numpy ndarray, and shape is a tuple (e.g., from
    array.shape). a is the input array shrunk such that its maximum
    dimensions are given by shape. If shape has more dimensions than
    array, the last dimensions of shape are fit.
    
    as, bs = shrink(a, b)
    
    If the second argument is also an array, both a and b are shrunk to
    the dimensions of each other. The input arrays must have the same
    number of dimensions, and the resulting arrays will have the same
    shape.
    Example
    -------
    
    >>> shrink(rand(10, 10), (5, 9, 18)).shape
    (9, 10)
    >>> map(shape, shrink(rand(10, 10, 10), rand(5, 9, 18)))        
    [(5, 9, 10), (5, 9, 10)]   
       
    """

    if isinstance(b, np.ndarray):
        if not len(a.shape) == len(b.shape):
            raise Exception, \
                  'input arrays must have the same number of dimensions'
        a = shrink(a,b.shape)
        b = shrink(b,a.shape)
        return (a, b)

    if isinstance(b, int):
        b = (b,)

    if len(a.shape) == 1:                # 1D array is a special case
        dim = b[-1]
        while a.shape[0] > dim:          # only shrink a
#            if (dim - a.shape[0]) >= 2:  # trim off edges evenly
            if (a.shape[0] - dim) >= 2:
                a = a[1:-1]
            else:                        # or average adjacent cells
                a = 0.5*(a[1:] + a[:-1])
    else:
        for dim_idx in range(-(len(a.shape)),0):
            dim = b[dim_idx]
            a = a.swapaxes(0,dim_idx)        # put working dim first
            while a.shape[0] > dim:          # only shrink a
                if (a.shape[0] - dim) >= 2:  # trim off edges evenly
                    a = a[1:-1,:]
                if (a.shape[0] - dim) == 1:  # or average adjacent cells
                    a = 0.5*(a[1:,:] + a[:-1,:])
            a = a.swapaxes(0,dim_idx)        # swap working dim back

    return a

def rot2d(x, y, ang):
    '''rotate vectors by geometric angle'''
    xr = x*np.cos(ang) - y*np.sin(ang)
    yr = x*np.sin(ang) + y*np.cos(ang)
    return xr, yr

nc = netCDF4.Dataset(url)
mask = nc.variables['mask_rho'][:]
lon_rho = nc.variables['lon_rho'][:]
lat_rho = nc.variables['lat_rho'][:]
time = nc.variables['time'][:]

dt = []
for i in range(len(time)):
    dt.append(datetime(2013,5,19,12,0,0)+timedelta(hours=time[i]))

u = nc.variables['u'][tidx, -1, :, :]
v = nc.variables['v'][tidx, -1, :, :]

u = shrink(u, mask[1:-1, 1:-1].shape)
v = shrink(v, mask[1:-1, 1:-1].shape)

lon_c = lon_rho[1:-1, 1:-1]
lat_c = lat_rho[1:-1, 1:-1]

p = plt.figure()
ax = p.add_subplot(111)
dmap = Basemap(projection='cyl',llcrnrlat=lat_c[0,0]-0.01,urcrnrlat=lat_c[-1,-1]+0.01,
               llcrnrlon=lon_c[0,0]-0.01,urcrnrlon=lon_rho[-1,-1]+1,resolution='h')
dmap.drawparallels(np.arange(int(lat_c[0,0]),int(lat_c[-1,-1])+1,1),labels=[1,0,0,0])
dmap.drawmeridians(np.arange(int(lon_c[0,0]),int(lon_rho[-1,-1])+1,1),labels=[0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='green')
dmap.drawmapboundary()
q = ax.quiver(lon_c[::isub,::isub], lat_c[::isub,::isub], u[::isub,::isub], v[::isub,::isub],
        scale=1.0/scale, pivot='middle', zorder=1e35, width=0.003, color='blue')
ax.quiverkey(q, 0.85, 0.07, 1.0, label=r'1 m s$^{-1}$', coordinates='figure')
plt.show()