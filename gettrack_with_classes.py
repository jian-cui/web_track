import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import jmath, jata
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import path
from datetime import timedelta
from conversions import dm2dd

class figure_with_basemap(mpl.figure.Figure):
    def __init__(self,lonsize,latsize,axes_num=1,interval_lon=1,interval_lat=1):
        '''
        draw the Basemap, set the axes num in the figure
        '''
        super(figure_with_basemap, self).__init__()
        self.lonsize, self.latsize = lonsize, latsize
        # self.fig = plt.figure()
        line_num = jmath.smallest_multpr(2,axes_num)
        if line_num == 1:
            column_num = 1
        else:
            column_num = 2
        self.ax = plt.subplot(line_num,column_num,1)
        self.dmap = Basemap(projection='cyl',
                            llcrnrlat=min(self.latsize)-0.01,
                            urcrnrlat=max(self.latsize)+0.01,
                            llcrnrlon=min(self.lonsize)-0.01,
                            urcrnrlon=max(self.lonsize)+0.01,
                            resolution='h',ax=self.ax)
        self.dmap.drawparallels(np.arange(int(min(self.latsize)),
                                          int(max(latsize))+1,interval_lat),
                                labels=[1,0,0,0])
        self.dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                                          int(max(lonsize))+1,interval_lon),
                                labels=[0,0,0,1])
        self.dmap.drawcoastlines()
        self.dmap.fillcontinents(color='grey')
        self.dmap.drawmapboundary()
        # self.cid = self.canvas.mpl_connect('button_press_event',
                                           # self.on_left_button_down)
    def on_left_button_down(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            print 'You clicked: %f, %f' % (x, y)
    def getSize(self):
        return self.lonsize, self.latsize
#    def setSize(self, size):
#        self.lonsize, self.latsize = size
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
        """Return tuple of indices for i,j that will completely cover the specified bounding box.     
        i = bbox2ij(lon,lat,bbox)
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
        print 'path: ', p
        print 'lons: ', lons
        points = np.vstack((lons.flatten(),lats.flatten())).T
        tshape = np.shape(lons)
        print tshape
#        inside = p.contains_points(points).reshape((n,m))
        inside = []
        for i in range(len(points)):
            inside.append(p.contains_point(points[i]))
        inside = np.array(inside, dtype=bool).reshape(tshape)
        print 'inside', inside
#        ii,jj = np.meshgrid(xrange(m),xrange(n))
        index = np.where(inside==True)
        print 'index: ', index
        if not index[0].tolist():          # bbox covers no area
            # print 'out of range.'
            # i0,i1,j0,j1 = 10000,10000,10000,10000
            raise Exception('no points in this area')
        else:
            # points_covered = [point[index[i]] for i in range(len(index))]
            # for i in range(len(index)):
                # p.append(point[index[i])
            # i0,i1,j0,j1 = min(index[1]),max(index[1]),min(index[0]),max(index[0])
            return index                
    def nearest_point_index(self, lon, lat, lons, lats, length=(1, 1)):
        '''
        Return the index of the nearest rho point.
        lon, lat: the coordiation of original point, float
        lats, lons: the coordiation of points want to be calculated.
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
        print 'index', len(index), index
        lon_covered = lons[index]
        lat_covered = lats[index]
        # lon_covered = np.array([lons[i] for i in index])
        # lat_covered = np.array([lats[i] for i in index])
        cp = np.cos(lat_covered*np.pi/180.)
        dx = (lon-lon_covered)*cp
        dy = lat-lat_covered
        dist = dx*dx+dy*dy
        print 'dist', len(dist),dist
        i = np.argmin(dist)
        findex = [j[i] for j in index]
        print findex
        return findex, dist[i]
    def waternode(self, timeperiod, data):
        pass
        
class water_roms(water):
    def __init__(self, startpoint,days):
        self.dataloc = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd'
        self.startpoint = startpoint
        self.days = days
    def get_data(self):
        self.data = jata.get_nc_data(self.dataloc, 'lon_rho', 'lat_rho',
                                                   'mask_rho', 'time', 'u', 'v')
        return self.data
    def waternode(self, data):
        lon, lat = self.startpoint[0], self.startpoint[1]
        # lon_nodes, lat_nodes = [], []
        data = self.get_data()
        nodes = dict(lon=[lon], lat=[lat])
        # nodes = dict(lon=[], lat=[])
        mask = self.data['mask_rho'][:]
        lon_rho = self.data['lon_rho'][:]
        lat_rho = self.data['lat_rho'][:]
        u = self.data['u'][:,-1]
        v = self.data['v'][:,-1]
        lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        start, end = u.shape[0]-self.days, u.shape[0]
        for i in range(start, end):
            # nodes['lon'].append(lon)
            # nodes['lat'].append(lat)
            u_t = jata.shrink(u[i], mask[1:,1:].shape)
            v_t = jata.shrink(v[i], mask[1:,1:].shape)
            index,nearestdistance = self.nearest_point_index(lon, lat, lons, lats)
            dx = 24*60*60*float(u_t[index[0],index[1]])
            dy = 24*60*60*float(v_t[index[0],index[1]])
            lon = lon + dx/(111111*np.cos(lat*np.pi/180))
            lat = lat + dy/111111
            nodes['lon'].append(lon)
            nodes['lat'].append(lat)
        return nodes

class water_fvcom(water):
    def __init__(self, modelname, days):
        '''
        starttime: datetime.datetime()
        '''
        self.modelname = modelname
        self.days = days
        # self.startpoint = startpoint
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
        days_datetime = timedelta(days=self.days)
        datakeys = ('u', 'v', 'lon', 'lat', 'lonc', 'latc', 'siglay', 'h')
        if self.modelname is '30yr':
            self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
                           ','.join(datakeys)
            yearnum = starttime.year-1981
            standardtime = datetime.strptime(str(starttime.year)+'-01-01 00:00:00',
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
            index1 = (period.seconds)/60/60
            index2 = index1+24*days_int
        else:
            raise Exception('Please use right model')
        return self.dataloc, [index1, index2]
    def get_data(self, dataloc):
        self.data = jata.get_nc_data(dataloc, 'lon', 'lat', 'latc', 'lonc',
                                     'u', 'v', 'siglay', 'h')
        return self.data
    def waternode(self, lon, lat, depth, start, end, data):
        '''
        start, end: indices of some period
        data: a dict that has 'u' and 'v'
        '''
        lonc,latc = data['lonc'][:], data['latc'][:]
        lonv, latv = data['lon'][:], data['lat'][:]
        h = data['h'][:]
        siglay = data['siglay'][:]
        if lon>90:
            lon, lat = dm2dd(lon, lat)
        nodes = dict(lon_nodes=[], lat_nodes=[])
        kf,distanceF = self.nearest_point_index(lon,lat,lonc,latc)
        kv,distanceV = self.nearest_point_index(lon,lat,lonv,latv)
        if h[kv] < 0:
            sys.exit('Sorry, your position is on land, please try another point')
        depth_total = siglay[:,kv]*h[kv]
        ###############layer###########################
        layer = np.argmin(abs(depth_total-depth))
        print start, end, layer, kf
        m = 0
        for i in range(start,end):
            m += 1
            print 'm:', m
            print data['u'][0,0,0]
            # u_t = np.array(data['u'])[i,layer,kf]
            # v_t = np.array(data['v'])[i,layer,kf]
            u_t = data['u'][i,layer,kf[0]]
            v_t = data['v'][i,layer,kf[0]]
            dx = 60*60*u_t
            dy = 60*60*v_t
            print 'dx',dx
            lon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            lat = lat + dy/111111
            nodes['lon_nodes'].append(lon)
            nodes['lat_nodes'].append(lat)
            print 'nodes', nodes
            kf, distanceF = self.nearest_point_index(lon, lat, lonc, latc)
            print 'kf',kf
            kv, distanceV = self.nearest_point_index(lon, lat, lonv, latv)
            depth_total = siglay[:,kv]*h[kv]
            if distanceV>=.3:
                if i==start:
                    print 'Sorry, your start position is NOT in the model domain'
                    break
        return nodes

class water_drifter(water):
    def __init__(self, drifter_id=None, starttime=None):
        self.dataloc = "/net/home3/ocn/jmanning/py/jc/web_track/drift_tcs_2013_1.dat"
        self.drifter_id = drifter_id
        self.starttime = starttime
    def waternode(self):
        # self.drifter_id = jata.input_with_default('drifter ID', 139420691)
        # self.starttime = datetime(year=2013, month=9, day=29, hour=11,minute=46)
        nodes = jata.data_extracted(self.dataloc, self.drifter_id, self.starttime)
        return nodes


modelname = 'FVCOM'

if modelname is 'drifter':
    starttime = datetime(year=2013, month=9, day=29, hour=11, minute=46)
    drifter_id = jata.input_with_default('drifter_id', 139420691)
    # dataloc = "/net/home3/ocn/jmanning/py/jc/web_track/drift_tcs_2013_1.dat"
    
    drifter = water_drifter(drifter_id, starttime)
    nodes = drifter.waternode()
    lonsize = min(nodes['lon'])-1, max(nodes['lon'])+1
    latsize = min(nodes['lat'])-1, max(nodes['lat'])+1
    fig = figure_with_basemap(lonsize, latsize)
    # fig.basemap(lonsize, latsize)
    plt.plot(nodes['lon'], nodes['lat'],'ro-')
    plt.show()
elif modelname is 'ROMS':
    # dataloc = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd'
    startpoint = (-73, 38.0)  #point wanted to be forecast
    days = 6                  #forecast 3 days later, and show [days-3] days before
    isub = 3                  #interval of arrow of water speed
    scale = 0.03              #
    tidx = -1                 #layer. -1 is the last one.
    
    water_roms = water_roms(startpoint, days)
    data = water_roms.get_data()
    nodes = water_roms.waternode(data)
    lonc = data['lon_rho'][1:-1, 1:-1]
    latc = data['lat_rho'][1:-1, 1:-1]
    u = data['u'][:, -1][tidx,:,:]
    v = data['v'][:, -1][tidx,:,:]
    u = jata.shrink(u, data['mask_rho'][1:-1, 1:-1].shape)
    v = jata.shrink(v, data['mask_rho'][1:-1, 1:-1].shape)
    lonsize = min(nodes['lon'])-1, max(nodes['lon'])+1
    latsize = min(nodes['lat'])-1, max(nodes['lat'])+1
    fig = figure_with_basemap(lonsize, latsize)
    fig.ax.quiver(lonc[::isub,::isub], latc[::isub,::isub],
                  u[::isub,::isub], v[::isub,::isub],
                  scale=1.0/scale, pivot='middle',
                  zorder=1e35, width=0.003,color='blue')
    plt.plot(nodes['lon'], nodes['lat'], 'ro-')
    plt.show()
elif modelname is 'FVCOM':
    model = 'massbay'
    days = 2
    #when you choose '30yr' model, please keep
    #starttime before 2010-12-31 after 1978.
    starttime = '2014-02-13 13:40:00'
    lon = -70.718466
    lat = 40.844644
    # lon = float(jata.input_with_default('lon', 7031.8486))
    # lat = float(jata.input_with_default('lat', 3934.4644))
    # starttime = jata.input_with_default('TIME','2014-02-07 13:40:00')
    starttime = datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
    depth = -3
    
    water_fvcom = water_fvcom(model, days)
    dataloc, index = water_fvcom.get_interval(starttime)
    data = water_fvcom.get_data(dataloc)
    nodes = water_fvcom.waternode(lon, lat, depth, index[0], index[1], data)
    lonsize = min(nodes['lon_nodes'])-1, max(nodes['lon_nodes'])+1
    latsize = min(nodes['lat_nodes'])-1, max(nodes['lat_nodes'])+1
    fig = figure_with_basemap(lonsize, latsize)
    fig.ax.plot(nodes['lon_nodes'], nodes['lat_nodes'], 'ro-')
    plt.show()

drifter_id = jata.input_with_default('drifter_id', 139420691)

drifter = water_drifter(drifter_id, starttime)
nodes = drifter.waternode()

startpoint = nodes['lon'][0], nodes['lat'][0]
