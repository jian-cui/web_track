import numpy as np
import matplotlib.pyplot as plt
import jata
from mpl_toolkits.basemap import Basemap

url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?lat[0:1:48450],latc[0:1:90414],lon[0:1:48450],lonc[0:1:90414]'
data = jata.get_nc_data(url, 'lon', 'lat', 'lonc', 'latc')

lonsize = [-76, -55]
latsize = [35, 46.5]
interval_lat, interval_lon = 0.5, 0.5
fig = plt.figure()
ax = plt.subplot(111)
dmap = Basemap(projection='cyl',
               llcrnrlat=min(latsize)-0.01,
               urcrnrlat=max(latsize)+0.01,
               llcrnrlon=min(lonsize)-0.01,
               urcrnrlon=max(lonsize)+0.01,
               resolution='h',ax=ax)
dmap.drawparallels(np.arange(int(min(latsize)),
                             int(max(latsize))+1,interval_lat),
                   labels=[1,0,0,0])
dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                             int(max(lonsize))+1,interval_lon),
                   labels=[0,0,0,1])
dmap.drawcoastlines()
dmap.fillcontinents(color='grey')
dmap.drawmapboundary()

# ax.plot(data['lon'], data['lat'], '.', label='nodal')
ax.plot(data['lonc'], data['latc'], '.', label='zonal')
plt.legend()
plt.show()
