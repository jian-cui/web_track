import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import jata
import math
from datetime import datetime, timedelta
import netCDF4
from matplotlib import path
class water(object):
    def __init__(self, startpoint):
        '''
        get startpoint of water, and the location of datafile.
        startpoint = [25,45]
        '''
        self.startpoint = startpoint
    def get_data(self, url):
        pass
    def bbox2ij(self, lons, lats, bbox):
        """
        Return tuple of indices of points that are completely covered by the 
        specific boundary box.
        i = bbox2ij(lon,lat,bbox)
        lons,lats = 2D arrays (list) that are the target of the subset, type: np.ndarray
        bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]
    
        Example
        -------  
        >>> i0,i1,j0,j1 = bbox2ij(lat_rho,lon_rho,[-71, -63., 39., 46])
        >>> h_subset = nc.variables['h'][j0:j1,i0:i1]       
        """
        bbox = np.array(bbox)
        mypath = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
        p = path.Path(mypath)
        points = np.vstack((lons.flatten(),lats.flatten())).T
        tshape = np.shape(lons)
        # inside = p.contains_points(points).reshape((n,m))
        inside = []
        for i in range(len(points)):
            inside.append(p.contains_point(points[i]))
        inside = np.array(inside, dtype=bool).reshape(tshape)
        # ii,jj = np.meshgrid(xrange(m),xrange(n))
        index = np.where(inside==True)
        if not index[0].tolist():          # bbox covers no area
            raise Exception('no points in this area')
        else:
            # points_covered = [point[index[i]] for i in range(len(index))]
            # for i in range(len(index)):
                # p.append(point[index[i])
            # i0,i1,j0,j1 = min(index[1]),max(index[1]),min(index[0]),max(index[0])
            return index
    def nearest_point_index(self, lon, lat, lons, lats, length=(1, 1),num=4):
        '''
        Return the index of the nearest rho point.
        lon, lat: the coordinate of start point, float
        lats, lons: the coordinate of points to be calculated.
        length: the boundary box.
        '''
        bbox = [lon-length[0], lon+length[0], lat-length[1], lat+length[1]]
        # i0, i1, j0, j1 = self.bbox2ij(lons, lats, bbox)
        # lon_covered = lons[j0:j1+1, i0:i1+1]
        # lat_covered = lats[j0:j1+1, i0:i1+1]
        # temp = np.arange((j1+1-j0)*(i1+1-i0)).reshape((j1+1-j0, i1+1-i0))
        # cp = np.cos(lat_covered*np.pi/180.)
        # dx=(lon-lon_covered)*cp
        # dy=lat-lat_covered
        # dist=dx*dx+dy*dy
        # i=np.argmin(dist)
        # # index = np.argwhere(temp=np.argmin(dist))
        # index = np.where(temp==i)
        # min_dist=np.sqrt(dist[index])
        # return index[0]+j0, index[1]+i0
        index = self.bbox2ij(lons, lats, bbox)
        lon_covered = lons[index]
        lat_covered = lats[index]
        # if len(lat_covered) < num:
            # raise ValueError('not enough points in the bbox')
        # lon_covered = np.array([lons[i] for i in index])
        # lat_covered = np.array([lats[i] for i in index])
        cp = np.cos(lat_covered*np.pi/180.)
        dx = (lon-lon_covered)*cp
        dy = lat-lat_covered
        dist = dx*dx+dy*dy
        
        # get several nearest points
        dist_sort = np.sort(dist)[0:9]
        findex = np.where(dist==dist_sort[0])
        lists = [[]] * len(findex)
        for i in range(len(findex)):
            lists[i] = findex[i]
        if num > 1:
            for j in range(1,num):
                t = np.where(dist==dist_sort[j])
                for i in range(len(findex)):
                     lists[i] = np.append(lists[i], t[i])
        indx = [i[lists] for i in index]
        return indx, dist_sort[0:num]
        '''
        # for only one point returned
        mindist = np.argmin(dist)
        indx = [i[mindist] for i in index]
        return indx, dist[mindist]
        '''
    def waternode(self, timeperiod, data):
        pass
class temp(water):
    def __init__(self):
        pass
    def get_url(self, starttime, endtime):
        url_oceantime = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?ocean_time[0:1:69911]'
        data_oceantime = netCDF4.Dataset(url_oceantime)
        t1 = (starttime - datetime(2006,01,01)).total_seconds()
        t2 = (endtime - datetime(2006,01,01)).total_seconds()
        index1 = self.__closest_num(t1,data_oceantime.variables['ocean_time'][:])
        index2 = self.__closest_num(t2,data_oceantime.variables['ocean_time'][:])
        url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?s_rho[0:1:35],h[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],temp[{0}:1:{1}][0:1:35][0:1:81][0:1:129],ocean_time[{0}:1:{1}]'
        url = url.format(index1, index2)
        return url
    def __closest_num(self, num, numlist, i=0):
        '''
        Return index of the closest number in the list
        '''
        index1, index2 = 0, len(numlist)
        indx = int(index2/2)
        if not numlist[0] < num < numlist[-1]:
            raise Exception('{0} is not in {1}'.format(str(num), str(numlist)))
        if index2 == 2:
            l1, l2 = num-numlist[0], numlist[-1]-num
            if l1 < l2:
                i = i
            else:
                i = i+1
        elif num == numlist[indx]:
            i = i + indx
        elif num > numlist[indx]:
            i = self.__closest_num(num, numlist[indx:],
                              i=i+indx)
        elif num < numlist[indx]:
            i = self.__closest_num(num, numlist[0:indx+1], i=i)
        return i
    def get_data(self, url):
        data = jata.get_nc_data(url, 'h', 'lat_rho', 'lon_rho', 'temp', 's_rho','ocean_time')
        return data
    def templine(self, lon, lat, url):
        data = self.get_data(url)
        lons = data['lon_rho'][:]
        lats = data['lat_rho'][:]
        index, d = self.nearest_point_index(lon, lat, lons, lats)
        # depth_layers = data['h'][index[0][0]][index[1][0]]*data['s_rho']
        # layer = np.argmin(abs(depth_layers-depth))
        layer = -1
        temp = data['temp'][:, layer, index[0][0], index[1][0]]
        temptime = []
        for i in range(0, len(data['ocean_time'][:])):
            dt = datetime(2006,1,1)+timedelta(seconds=data['ocean_time'][i])
            temptime = np.append(temptime, dt)
            print temptime
        return temp, temptime
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
    lon, lat = event.xdata, event.ydata
    if lon is None:
        print 'Sorry, please click another point'
    else:
        print "You click: ", lon, lat
        tempobj = temp()
        url = tempobj.get_url(starttime, endtime)
        dtemp, dtime = tempobj.templine(lon, lat, url)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(dtime, dtemp)
        plt.title('lon:{0},lat:{1},From:{2}'.format(lon, lat, starttime))
        plt.show()
    '''
    tempobj = temp()
    url = tempobj.get_url(starttime, endtime)
    dtemp, dtime = tempobj.templine(x, y, url)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dtime, dtemp)
    plt.show()
    '''
starttime, endtime= datetime(2006,05,19), datetime(2006,05,21)
depth = -1
url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?h[0:1:81][0:1:129],temp[0:1:307][0:1:35][0:1:81][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'h', 'temp')
lonsize = np.amin(data['lon_rho'][:])-1, np.amax(data['lon_rho'][:])+1
latsize = np.amin(data['lat_rho'][:])-1, np.amax(data['lat_rho'][:])+1
'''
fig = plt.figure()
ax = fig.add_subplot(111)
x = range(1, 309)
ax.plot(range(1, 309), data['temp'][:, 0, 25, 35])
# cid = fig.canvas.mpl_connect('button_press_event', left_button_down)
# plt.xticks(x, range(5,308,50))
plt.show()
'''
fig = plt.figure()
ax = plt.subplot(111)
cid = fig.canvas.mpl_connect('button_press_event', left_button_down)
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
plt.title('ROMS Bttom temp of {0}'.format(starttime))
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
