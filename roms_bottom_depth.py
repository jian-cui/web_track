import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import jata
import math
def vertical_point(p1, p2, p0):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p0[0], p0[1]
    x = ((x2-x1)*(y2-y1)*(y3-y1)+(x2-x1)**2*x3+(y2-y1)**2*x1)/\
        ((y2-y1)**2+(x2-x1)**2)
    y = ((y2-y1)*x-x1*y2+x2*y1)/\
        (x2-x1)
    return x, y
def value_on_proportion(p1, p2, p0, v1, v2):
    '''
    p1, p2, p0 are on the same line, v1, v2 are the value of p1, p2,
    calculate and return the value of p0
    '''
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x0, y0 = p0[0], p0[1]
    dist01 = math.sqrt((y0-y1)**2+(x0-x1)**2)
    dist12 = math.sqrt((y1-y2)**2+(x1-x2)**2)
    v3 = (v2-v1)*dist01/dist12+v1
    return v3
def left_button_down(event):
    x, y = event.xdata, event.ydata
    print x, y
def rk_4(func, func(0, 0), h):
    while n is not 0:
        f[n] = rk_r()
        f[n+1] = f[n]+h*(k1+2k2+2K3+k4)/6
        k1 = func(x[n], y[n])
        
url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?h[0:1:81][0:1:129],temp[0:1:307][0:1:35][0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'h', 'temp')
lonsize = np.amin(data['lon_rho'][:])-1, np.amax(data['lon_rho'][:])+1
latsize = np.amin(data['lat_rho'][:])-1, np.amax(data['lat_rho'][:])+1


fig = plt.figure()
ax = plt.subplot(111)
fig.canvas.mpl_connect('button_press_event', left_button_down)
dmap = Basemap(projection = 'cyl',
               llcrnrlat = min(latsize)-0.01,
               urcrnrlat = max(latsize)+0.01,
               llcrnrlon = min(lonsize)-0.01,
               urcrnrlon = max(lonsize)+0.01,
               resolution = 'h', ax = ax)
dmap.drawparallels(np.arange(int(min(latsize)), int(max(latsize))+1, 0.5),
                   labels = [1,0,0,0])
dmap.drawmeridians(np.arange(int(min(lonsize)), int(max(lonsize))+1, 0.5),
                   labels = [0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='grey')
dmap.drawmapboundary()

cs = plt.contourf(data['lon_rho'], data['lat_rho'], data['h'], range(0,400),
                  extend='both')
plt.colorbar()
# plt.clabel(cs, inline=0, fontsize=10)
plt.show()
'''
# fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True,sharey=True)
fig = plt.figure()
ax = plt.subplot(211)
dmap = Basemap(projection = 'cyl',
               llcrnrlat = min(latsize)-0.01,
               urcrnrlat = max(latsize)+0.01,
               llcrnrlon = min(lonsize)-0.01,
               urcrnrlon = max(lonsize)+0.01,
               resolution = 'h', ax = ax)
dmap.drawparallels(np.arange(int(min(latsize)), int(max(latsize))+1, 0.5),
                   labels = [1,0,0,0])
dmap.drawmeridians(np.arange(int(min(lonsize)), int(max(lonsize))+1, 0.5),
                   labels = [0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='grey')
dmap.drawmapboundary()
ax.set_title('36th layer')
cs = ax.contourf(data['lon_rho'], data['lat_rho'], data['temp'][296,35], 100)
plt.colorbar(cs)

ax2 = plt.subplot(212)
dmap = Basemap(projection = 'cyl',
               llcrnrlat = min(latsize)-0.01,
               urcrnrlat = max(latsize)+0.01,
               llcrnrlon = min(lonsize)-0.01,
               urcrnrlon = max(lonsize)+0.01,
               resolution = 'h', ax = ax2)
dmap.drawparallels(np.arange(int(min(latsize)), int(max(latsize))+1, 0.5),
                   labels = [1,0,0,0])
dmap.drawmeridians(np.arange(int(min(lonsize)), int(max(lonsize))+1, 0.5),
                   labels = [0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='grey')
dmap.drawmapboundary()
ax2.set_title('1st layer')
cs2 = ax2.contourf(data['lon_rho'], data['lat_rho'], data['temp'][296,0], 100)
# cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(cax=cax, **kw)
# fig.subplots_adjust()
# cax = fig.add_axes()
# fig.colorbar(cs2)
plt.colorbar(cs2)
plt.show()
'''
