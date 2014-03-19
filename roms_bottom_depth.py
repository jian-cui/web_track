import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import jata

# url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?h[0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?h[0:1:81][0:1:129],temp[0:1:296][0:1:35][0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'h', 'temp')
lonsize = np.amin(data['lon_rho'][:])-1, np.amax(data['lon_rho'][:])+1
latsize = np.amin(data['lat_rho'][:])-1, np.amax(data['lat_rho'][:])+1


'''
fig = plt.figure()
ax = plt.subplot(111)
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
