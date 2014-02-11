import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import jmath, jata
class figure_with_basemap:
    def __init__(self,lonsize,latsize,axes_num=1,interval_lon=1,interval_lat=1):
        '''
        draw the Basemap, set the axes num in the figure
        '''
#        super(add_figure, self).__init__()
        self.lonsize, self.latsize = lonsize, latsize
        self.fig = plt.figure()
        line_num = jmath.smallest_multpr(2,axes_num)
        if line_num == 1:
            column_num = 1
        else:
            column_num = 2
        self.ax = plt.subplot(line_num,column_num,1)
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self.on_left_button_down)
        self.dmap = Basemap(projection='cyl',
                            llcrnrlat=min(self.latsize)-0.01,
                            urcrnrlat=max(self.latsize)+0.01,
                            llcrnrlon=min(self.lonsize)-0.01,
                            urcrnrlon=max(self.lonsize)+0.01,
                            resolution='h',ax=ax)
        self.dmap.drawparallels(np.arange(int(min(self.latsize)),
                                          int(max(latsize))+1,interval_lat),
                                labels=[1,0,0,0])
        self.dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                                          int(max(lonsize))+1,interval_lon),
                                labels=[0,0,0,1])
        self.dmap.drawcoastlines()
        self.dmap.fillcontinents(color='grey')
        self.dmap.drawmapboundary()
    def on_left_button_down(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            print 'You clicked: %f, %f' % (x, y)
    def getSize(self):
        return self.lonsize, self.latsize
#    def setSize(self, size):
#        self.lonsize, self.latsize = size
    def show():
        self.fig.show()
    size = property(getSize)
class water(object):
    def __init__(self, startpoint, dataloc, modelname):
        '''
        get startpoint of water, and the location of datafile.
        startpoint = [25,45]
        dataloc = 'http://getfile.com/file.nc'
        or
        dataloc = '/net/usr/datafile.nc'
        '''
        self.modelname = modelname
        self.startpoint = startpoint
        self.dataloc = dataloc
    def get_data(self, dataloc):
        pass
    def bbox2ij(self, lons, lats, bbox):
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
#        print mypath
        p = path.Path(mypath)
        points = np.vstack((lons.flatten(),lats.flatten())).T   
        n,m = np.shape(lons)
#        inside = p.contains_points(points).reshape((n,m))
        inside = []
        for i in range(len(points)):
            inside.append(p.contains_point(points[i]))
        inside = np.array(inside, dtype=bool).reshape((n, m))
#        ii,jj = np.meshgrid(xrange(m),xrange(n))
        index = np.where(inside==True)
        if not index[0].tolist():          # bbox covers no area
            # print 'out of range.'
            # i0,i1,j0,j1 = 10000,10000,10000,10000
            raise(Exception, 'no points in this area')
        else:
            i0,i1,j0,j1 = min(index[1]),max(index[1]),min(index[0]),max(index[0])
        return i0, i1, j0, j1
    def nearest_point_index(self, lon, lat, lons, lats, length=(1, 1)):
        '''
        Return the index of the nearest rho point.
        lon, lat: the coordiation of original point, float
        lats, lons: the coordiation of points want to be calculated.
        length: the boundary box.
        '''
        bbox = [lon-length[0], lon+length[0], lat-length[1], lat+length[1]]
        i0, i1, j0, j1 = self.bbox2ij(lons, lats, bbox)
        lon_covered = lons[j0:j1+1, i0:i1+1]
        lat_covered = lats[j0:j1+1, i0:i1+1]
        temp = np.arange((j1+1-j0)*(i1+1-i0)).reshape((j1+1-j0, i1+1-i0))
        cp = np.cos(lat_covered*np.pi/180.)
        dx=(lon-lon_covered)*cp
        dy=lat-lat_covered
        dist=dx*dx+dy*dy
        i=np.argmin(dist)
        index = np.argwhere(temp=np.argmin(dist))
        index = np.where(temp==i)
        min_dist=np.sqrt(dist[index])
        return index[0]+j0, index[1]+i0
    def waternode(self, timeperiod, data):
        pass
        
class water_roms(water):
    def __init__(self, dataloc, startpoint, modelname):
        self.startpoint = startpoint
        self.dataloc = dataloc
        self.startpoint = startpoint
    def get_data(self):
        self.data = jata.get_nc_data(self.dataloc, 'mask_rho', 'lon_rho',
                                                   'lat_rho', 'time', 'u', 'v')
        return self.data
    def waternode(self, timeperiod, data):
        lon, lat = self.startpoint[0], self.startpointt[1]
        lon_nodes, lat_nodes = [], []
        mask = data['mask_rho'][:]
        lon_rho = data['lon_rho'][:]
        lat_rho = data['lat_rho'][:]
        u = data['u'][:,-1]
        v = data['v'][:,-1]
        lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        start, end = u.shape[0]-days, u.shape[0]
        for i in range(start, end):
            lon_node.append(lon)
            lat_node.append(lat)
            u_t = jata.shrink(u[i], mask[1:,1:].shape)
            v_t = jata.shrink(v[i], mask[1:,1:].shape)
            index = nearest_point_index(lon, lat, lons, lats)
            dx = 24*60*60*u_t[index[0],index[1]]
            dy = 24*60*60*v_t[index[0],index[1]]
            lon = lon + dx/(111111*np.cos(lat*np.pi/180))
            lat = lat + dy/111111
        return lon_nodes, lat_nodes

class water_fvcom(water):
    def __init__(self, startpoint, starttime, modelname, days):
        '''
        starttime: datetime.datetime()
        '''
        self.modelname = modelname
        self.days = days
        self.startpoint = startpoint
        # data = ('lon', 'lat', 'lonc', 'latc', 'siglay', 'h')
        # if self.modelname is '30yr':
        #     self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
        #                    ','.join(data)
        # elif self.modelname is 'GOM3':
        #     self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?'+\
        #                    ','.join(data)
        # elif self.modelname is 'massbay':
        #     self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?'+\
        #                    ','.join(data)
        # else:
        #     raise Exception('Please use right model')
    def get_interval(self, starttime):
        '''
        starttime: datetime.datetime()
        '''
        days_int = self.days
        days_datetime = time.timedelta(days=self.days)
        datakeys = ('lon', 'lat', 'lonc', 'latc', 'siglay', 'h')
        if self.modelname is '30yr':
            self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
                           ','.join(datakeys)
            yearnum = starttime.year-1981
            standardtime = datatime.strptime(str(starttime.year)+'-01-01 00:00:00',
                                             '%Y-%m-%d %H:%M:%S')
            index1 = 26340+35112*(yearnum/4)+8772*(yearnum%4)+1+\
                      24*(starttime-standardtime).days
            index2 = index1+24*days_int
        elif self.modelname is 'GOM3':
            self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?'+\
                           ','.join(datakeys)
            period = (starttime+days_datetime)-(datetime.now()-timedelta(days=3))
            index1 = (period.seconds)/60/60
            index2 = index1 + 24*(days_int)
        elif self.modelname is 'massbay':
            self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?'+\
                           ','.join(datakeys)
            period = (starttime+days_datetime)-(datetime.now()-timedelta(days=3))
            index1 = (timeperiod.seconds)/60/60
            index2 = index1+24*days_int
        else:
            raise Exception('Please use right model')
    def get_data(self):
        self.data = jata.get_nc_data(self.dataloc, 'lon', 'lat', 'latc', 'lonc',
                                     'siglay', 'h', 'Times')
        return self.data
    def waternode(self, lon, lat, lonc, latc, lonv, latv, start, end, data):
        '''
        start, end: indices of some period
        data: a dict that has 'u' and 'v'
        '''
        if lon>90:
            lon, lat = dm2dd(lon, lat)
        nodes = dict(lon_nodes=[], lat_nodes=[])
        kf,distanceF = nearest_point_index(lon,lat,lonc,latc)
        kv,distanceV = nearest_point_index(lon,lat,lonv,latv)
        if h[kv] < 0:
            sys.exit('Sorry, your position is on land, please try another point')
        depth_total = siglay[:,kv]*h[kv]
        layer = np.argmin(abs(depthtotal-depth))
        for i in range(start,end):
            u_t = np.array(data['u'])[i,layer,kf]
            v_t = np.array(data['v'])[i,layer,kf]
            dx = 60*60*u_t
            dy = 60*60*v_t
            lon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            lat = lat + dy/111111
            nodes['lon_nodes'].append(lon)
            nodes['lat_nodes'].append(lat)
            kf, distanceF = nearest_point_index(lon, lat, lonc, latc)
            kv, distanceV = nearest_point_index(lon, lat, lonv, latv)
            depth_total = siglay[:,kv]*h[kv]
            if distanceV>=.3:
                if i==start:
                    print 'Sorry, your start position is NOT in the model domain'
                    break
        return nodes

class water_drifter(water):
    def __init__(self, dataloc,drifter_id=None, starttime=None):
        self.dataloc = dataloc
        self.startpoint = startpoint
        self.starttime = starttime
    def waternode(self):
        # self.drifter_id = jata.input_with_default('drifter ID', 139420691)
        # self.starttime = datetime(year=2013, month=9, day=29, hour=11,minute=46)
        nodes = data_extracted(dataloc, self.drifter_id, self.starttime)
        return nodes
