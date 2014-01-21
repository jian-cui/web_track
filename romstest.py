# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:17:40 2014

@author: jc
"""

import matplotlib.pyplot as plt
from matplotlib import path
import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as np
from datetime import datetime,timedelta

scale = 0.03
isub = 2
tidx = -1  # to input in which day you want to forecast(third day is -1, second day is -2,first day is -3)
url = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd'
lon = -75.80
lat = 33.75
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
def bbox2ij(lon, lat, bbox):
    """Return indices for i,j that will completely cover the specified bounding box.     
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox)
    lon,lat = 2D arrays that are the target of the subset, type: np.ndarray
    bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]

    Example
    -------  
    >>> i0,i1,j0,j1 = bbox2ij(lat_rho,lon_rho,[-71, -63., 39., 46])
    >>> h_subset = nc.variables['h'][j0:j1,i0:i1]       
    """
    bbox = np.array(bbox)
    mypath = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(),lat.flatten())).T   
    n,m = np.shape(lon)
#    inside = p.contains_points(points).reshape((n,m))
    inside = []
    for i in range(len(points)):
        inside.append(p.contains_point(points[i]))
    inside = np.array(inside, dtype=bool).reshape((n, m))
    ii,jj = np.meshgrid(xrange(m),xrange(n))
#    return min(ii[inside]),max(ii[inside]),min(jj[inside]),max(jj[inside])
#    return ii[inside], ii[inside].max(), jj[inside].min(), jj[inside].max()
    return np.min(ii[inside]),np.max(ii[inside]),np.min(jj[inside]),np.max(jj[inside])
def nearest_point_index(lon, lat, lats, lons, length=(0.5, 0.5)):
    '''
    Return the index of the nearest rho point.
    lon, lat: the coordiation of original point
    lats, lons: the coordiation of points want to be calculated.
    length: the boundary box.
    '''
    bbox = [lon-length[0], lon+length[0], lat-length[1], lat+length[1]]
    i0, i1, j0, j1 = bbox2ij(lons, lats, bbox)
    lon_covered = lons[j0:j1+1, i0:i1+1]
    lat_covered = lats[j0:j1+1, i0:i1+1]
    temp = np.arange((j1+1-j0)*(i1+1-i0)).reshape((j1+1-j0, i1+1-i0))
    cp = np.cos(lat_covered*np.pi/180.)
    dx=(lon-lon_covered)*cp
    dy=lat-lat_covered
    dist=dx*dx+dy*dy
    i=np.argmin(dist)
#    index = np.argwhere(temp=np.argmin(dist))
    index = np.where(temp==i)
    min_dist=np.sqrt(dist[i])
    return index[0]+j0, index[1]+i0
    
nc = netCDF4.Dataset(url)
mask = nc.variables['mask_rho'][:]
lon_rho = nc.variables['lon_rho'][:]
lat_rho = nc.variables['lat_rho'][:]
time = nc.variables['time'][:]
u = nc.variables['u'][:, -1]
v = nc.variables['v'][:, -1]
lons = shrink(lon_rho, mask[1:, 1:].shape)
lats = shrink(lat_rho, mask[1:, 1:].shape)
lon_p, lat_p = [], []
start, end = u.shape[0]-6, u.shape[0]
#dt = []
#for i in range(len(time)):
#    dt.append(datetime(2013,5,19,12,0,0)+timedelta(hours=time[i]))
for i in np.arange(start, end):
    lon_p.append(lon)
    lat_p.append(lat)
    u_t = shrink(u[i], mask[1:, 1:].shape)
    v_t = shrink(v[i], mask[1:, 1:].shape)
    index = nearest_point_index(lon, lat, lons, lats)
    dx = 24*60*60*u_t[index[0], index[1]]
    dy = 34*60*60*v_t[index[0], index[1]]
    lon = lon + dx/(111111*np.cos(lat*np.pi/180))
    lat = lat + dy/111111
    
#u = nc.variables['u'][tidx, -1, :, :]
#v = nc.variables['v'][tidx, -1, :, :]

#u = shrink(u, mask[1:-1, 1:-1].shape)
#v = shrink(v, mask[1:-1, 1:-1].shape)

#lon_c = lon_rho[1:-1, 1:-1]
#lat_c = lat_rho[1:-1, 1:-1]

p = plt.figure()
ax = p.add_subplot(111)
dmap = Basemap(projection='cyl',llcrnrlat=np.amin(lats)-0.01,urcrnrlat=np.amax(lats)+0.01,
               llcrnrlon=np.amin(lons)-0.01,urcrnrlon=np.amax(lons)+0.01,resolution='h')
dmap.drawparallels(np.arange(int(np.amin(lats)),int(np.amax(lats))+1,0.1),labels=[1,0,0,0])
dmap.drawmeridians(np.arange(int(np.amin(lons)),int(np.amax(lons))+1,0.1),labels=[0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='green')
dmap.drawmapboundary()
#q = ax.quiver(lon_c[::isub,::isub], lat_c[::isub,::isub], u[::isub,::isub], v[::isub,::isub],
#        scale=1.0/scale, pivot='middle', zorder=1e35, width=0.003, color='blue')
#ax.quiverkey(q, 0.85, 0.07, 1.0, label=r'1 m s$^{-1}$', coordinates='figure')
plt.plot(lon_p,lat_p,'ro-')
plt.show()