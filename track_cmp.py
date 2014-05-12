import sys
sys.path.append('../moj')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import Basemap
import jmath, jata
from datetime import datetime, timedelta
from matplotlib import path
from conversions import dm2dd
from getdata import getdrift
import calendar
import netCDF4
'''
class figure_with_basemap(mpl.figure.Figure):
    def __init__(self,lonsize,latsize,axes_num=1,interval_lon=0.5,interval_lat=0.5):
        # draw the Basemap, set the number of panels in the figure
        super(figure_with_basemap, self).__init__()
        self.lonsize, self.latsize = lonsize, latsize
        line_num = jmath.smallest_multpr(2,axes_num)
        if line_num == 1:
            column_num = 1
        else:
            column_num = 2
        self.ax = plt.subplot(line_num,column_num,1)
        self.dmap = Basemap(projection='cyl',
                            llcrnrlat=min(latsize)-0.01,
                            urcrnrlat=max(latsize)+0.01,
                            llcrnrlon=min(lonsize)-0.01,
                            urcrnrlon=max(lonsize)+0.01,
                            resolution='h',ax=self.ax)
        self.dmap.drawparallels(np.arange(int(min(latsize)),
                                          int(max(latsize))+1,interval_lat),
                                labels=[1,0,0,0])
        self.dmap.drawmeridians(np.arange(int(min(lonsize))-1,
                                          int(max(lonsize))+1,interval_lon),
                                labels=[0,0,0,1])
        self.dmap.drawcoastlines()
        self.dmap.fillcontinents(color='grey')
        self.dmap.drawmapboundary()
        # self.cid = self.canvas.mpl_connect('button_press_event',
        #                                    self.on_left_button_down)
    def on_left_button_down(self, event):
        # not necessary but here to demonstrate the action of clicking on the map
        if event.button == 1:
            x, y = event.xdata, event.ydata
            print 'You clicked: %f, %f' % (x, y)
    def getSize(self):
        return self.lonsize, self.latsize
    size = property(getSize)
'''
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
class water_roms(water):
    '''
    ####(2009.10.11, 2013.05.19):version1(old) 2009-2013
    ####(2013.05.19, present): version2(new) 2013-present
    (2006.01.01 01:00, 2014.1.1 00:00)
    '''
    def __init__(self):
        pass
        # self.startpoint = lon, lat
        # self.dataloc = self.get_url(starttime)
    def get_url(self, starttime, endtime):
        '''
        get url according to starttime and endtime.
        '''
        self.starttime = starttime
        # self.hours = int((endtime-starttime).total_seconds()/60/60) # get total hours
        # time_r = datetime(year=2006,month=1,day=9,hour=1,minute=0)
        url_oceantime = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?ocean_time[0:1:69911]'
        data_oceantime = netCDF4.Dataset(url_oceantime)
        t1 = (starttime - datetime(2006,01,01)).total_seconds()
        t2 = (endtime - datetime(2006,01,01)).total_seconds()
        index1 = self.__closest_num(t1,data_oceantime.variables['ocean_time'][:])
        index2 = self.__closest_num(t2,data_oceantime.variables['ocean_time'][:])
        # index1 = (starttime - time_r).total_seconds()/60/60
        # index2 = index1 + self.hours
        url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?h[0:1:81][0:1:129],s_rho[0:1:35],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
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
        '''
        return the data needed.
        url is from water_roms.get_url(starttime, endtime)
        '''
        data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'mask_rho','u', 'v', 'h', 's_rho')
        return data
    def waternode(self, lon, lat, depth, url):
        '''
        get the nodes of specific time period
        lon, lat: start point
        url: get from get_url(starttime, endtime)
        depth: 0~35, the 36th is the bottom.
        '''
        self.startpoint = lon, lat
        if type(url) is str:
            nodes = self.__waternode(lon, lat, depth, url)
        else: # case where there are two urls, one for start and one for stop time
            nodes = dict(lon=[self.startpoint[0]],lat=[self.startpoint[1]])
            for i in url:
                temp = self.__waternode(nodes['lon'][-1], nodes['lat'][-1], depth, i)
                nodes['lon'].extend(temp['lon'][1:])
                nodes['lat'].extend(temp['lat'][1:])
        return nodes # dictionary of lat and lon
    def __waternode(self, lon, lat, depth, url):
        '''
        return points
        '''
        data = self.get_data(url)
        nodes = dict(lon=lon, lat=lat)
        mask = data['mask_rho'][:]
        lon_rho = data['lon_rho'][:]
        lat_rho = data['lat_rho'][:]
        # lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        # lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        lons, lats = lon_rho[:-2, :-2], lat_rho[:-2, :-2]
        index, nearestdistance = self.nearest_point_index(lon,lat,lons,lats)
        depth_layers = data['h'][index[0][0]][index[1][0]]*data['s_rho']
        layer = np.argmin(abs(depth_layers+depth))
        u = data['u'][:,layer]
        v = data['v'][:,layer]
        # lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        # lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        for i in range(0, len(data['u'][:])):
            # u_t = jata.shrink(u[i], mask[1:,1:].shape)
            # v_t = jata.shrink(v[i], mask[1:,1:].shape)
            u_t = u[i][:-2, :]
            v_t = v[i][:,:-2]
            # index, nearestdistance = self.nearest_point_index(lon,lat,lons,lats)
            u_p = u_t[index[0][0]][index[1][0]]
            v_p = v_t[index[0][0]][index[1][0]]
            '''
            for ut, vt in zip(u_p, v_p):
                if ut:
                    break
            if not ut:
                # raise Exception('point hit the land')
                print 'point hit the land'
                break
            if not ut:
                print 'point hit the land'
                break
            u_p = u_t[index[0]][index[1]]
            v_p = v_t[index[0]][index[1]]
            '''
            if not u_p:
                print 'point hit the land'
                break
            dx = 60*60*float(u_p)
            dy = 60*60*float(v_p)
            lon = lon + dx/(111111*np.cos(lat*np.pi/180))
            lat = lat + dy/111111
            index, nearestdistance = self.nearest_point_index(lon,lat,lons,lats)
            nodes['lon'] = np.append(nodes['lon'],lon)
            nodes['lat'] = np.append(nodes['lat'],lat)
        return nodes
class water_fvcom(water):
    def __init__(self):
        self.modelname = 'GOM3'
    def get_url(self, starttime, endtime):
        '''
        get different url according to starttime and endtime.
        urls are monthly.
        '''
        self.hours = int((endtime-starttime).total_seconds()/60/60)
        if self.modelname is "30yr":
            url = []
            time1 = datetime(year=2011,month=1,day=1)      #all these datetime are made based on the model.
            time2 = datetime(year=2011,month=11,day=11)      #The model use different version data of different period.
            time3 = datetime(year=2013,month=05,day=9)
            time4 = datetime(year=2013,month=12,day=1)
            if endtime < time1:
                yearnum = starttime.year-1981
                standardtime = datetime.strptime(str(starttime.year)+'-01-01 00:00:00',
                                                 '%Y-%m-%d %H:%M:%S')
                index1 = int(26340+35112*(yearnum/4)+8772*(yearnum%4)+1+self.hours)
                index2 = index1 + self.hours
                furl = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?h[0:1:48450],lat[0:1:48450],latc[0:1:90414],lon[0:1:48450],lonc[0:1:90414],u[{0}:1:{1}][0:1:44][0:1:90414],v[{0}:1:{1}][0:1:44][0:1:90414],siglay'
                url.append(furl.format(index1, index2)) 
            elif time1 <= endtime < time2: # endtime is in GOM3_v11
                url.extend(self.__temp(starttime,endtime,time1,time2))
            elif time2 <= endtime < time3:  # endtime is in GOM3_v12
                url.extend(self.__temp(starttime,endtime,time2,time3))
            elif time3 <= endtime < time4:
                url.extend(self.__temp(starttime,endtime,time3,time4))
        elif self.modelname is "GOM3":
            url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?lon[0:1:51215],lat[0:1:51215],lonc[0:1:95721],latc[0:1:95721],siglay[0:1:39][0:1:51215],h[0:1:51215],u[{0}:1:{1}][0:1:39][0:1:95721],v[{0}:1:{1}][0:1:39][0:1:95721]'
            period = starttime-\
                     (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
            index1 = int(period.total_seconds()/60/60)
            print 'index1', index1
            index2 = index1 + self.hours
            url = url.format(index1, index2)
        elif self.modelname is "massbay":
            url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?lon[0:1:98431],lat[0:1:98431],lonc[0:1:165094],latc[0:1:165094],siglay[0:1:9][0:1:98431],h[0:1:98431],u[{0}:1:{1}][0:1:9][0:1:165094],v[{0}:1:{1}][0:1:9][0:1:165094]'
            period = starttime-\
                     (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
            index1 = int(period.total_seconds()/60/60)
            index2 = index1 + self.hours
            url = url.format(index1, index2)
        return url
    def __temp(self, starttime, endtime, time1, time2):
        if time1 <= endtime < time2:
            pass
        else:
            sys.exit('{0} not in the right period'.format(endtime))
        url = []
        if starttime >= time1:    #start time is from 2011.11.10 as v12
            if starttime.month == endtime.month:
                url.append(self.__url(starttime.year,starttime.month,
                                            [starttime.day,starttime.hour],
                                            [endtime.day,endtime.hour]))
            else:
                if starttime.year == endtime.year:
                    y = starttime.year
                    for i in range(starttime.month, endtime.month+1):
                        if i == starttime.month:
                            url.append(self.__url(y,i,
                                                  [starttime.day, starttime.hour],
                                                  [calendar.monthrange(y,i)[1],0]))
                        elif starttime.month < i < endtime.month:
                            url.append(self.__url(y,i,[1,0],
                                                  [calendar.monthrange(y,i)[1],0]))
                        elif i == endtime.month:
                            url.append(self.__url(y,i,[1,0],
                                                  [endtime.day,endtime.hour]))
                else:
                    for i in range(starttime.year, endtime.year+1):
                        if i == starttime.year:
                            url.extend(self.get_url(starttime,
                                               datetime(year=i,
                                                        month=12,day=31)))
                        elif i == endtime.year:
                            url.extend(self.get_url(datetime(year=i,month=1,day=1),
                                               endtime))
                        else:
                            url.extend(self.get_url(datetime(year=i,month=1,day=1),
                                               datetime(year=i,month=12,day=31)))
             
        else:
            url.extend(self.get_url(starttime,(time1-timedelta(minutes=1))))
            url.extend(self.get_url(time1,endtime))
        return url
    def __url(self, year, month, start_daytime, end_daytime):
        '''
        start_daytime,end_daytime: [day,hour]
        '''
        url_v11 = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM3_{0}/gom3v11_{0}{1}.nc?lon[0:1:48727],lat[0:1:48727],lonc[0:1:90997],latc[0:1:90997],h[0:1:48727],u[{2}:1:{3}][0:1:39][0:1:90997],v[{2}:1:{3}][0:1:39][0:1:90997],siglay[0:1:39][0:1:48727]'
        url_v12 = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM3_{0}/gom3v12_{0}{1}.nc?lon[0:1:48859],lat[0:1:48859],lonc[0:1:91257],latc[0:1:91257],h[0:1:48859],u[{2}:1:{3}][0:1:39][0:1:91257],v[{2}:1:{3}][0:1:39][0:1:91257],siglay[0:1:39][0:1:48859]'
        url_v13 = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM3_{0}/gom3v13_{0}{1}.nc?lon[0:1:51215],lat[0:1:51215],lonc[0:1:95721],latc[0:1:95721],h[0:1:51215],u[{2}:1:{3}][0:1:39][0:1:95721],v[{2}:1:{3}][0:1:39][0:1:95721],siglay[0:1:39][0:1:51215]'
        time1 = datetime(year=2011,month=1,day=1)      #all these datetime are made based on the model.
        time2 = datetime(year=2011,month=11,day=11)      #The model use different version data of different period.
        time3 = datetime(year=2013,month=05,day=9)
        time4 = datetime(year=2013,month=12,day=1)
        currenttime = datetime(year=year,month=month,day=start_daytime[0])
                                       
        if time1 <= currenttime < time2:
            version = '11'
        elif time2 <= currenttime < time3:
            version = '12'
        elif time3 <= currenttime < time4:
            version = '13'

        if year == 2011 and month == 11  and start_daytime[0] >10:
            start = str(24*(start_daytime[0]-1)+start_daytime[1]-240)
            end = str(24*(end_daytime[0]-1)+end_daytime[1]-240)
        elif year == 2013 and month == 5 and start_daytime[0] >8:
            start = str(24*(start_daytime[0]-1)+start_daytime[1]-192)
            end = str(24*(end_daytime[0]-1)+end_daytime[1]-192)
        else:
            start = str(24*(start_daytime[0]-1)+start_daytime[1])
            end = str(24*(end_daytime[0]-1)+end_daytime[1])
        year = str(year)
        month = '{0:02d}'.format(month)
        
        if version == '11':
            url = url_v11.format(year, month, start, end)
        elif version == '12':
            url = url_v12.format(year, month, start, end)
        elif version == '13':
            url = url_v13.format(year, month, start, end)
        return url
    def get_data(self,url):
        self.data = jata.get_nc_data(url,'lon','lat','latc','lonc',
                                     'u','v','siglay','h')
        return self.data
    def waternode(self, lon, lat, depth, url):
        if type(url) is str:
            nodes = dict(lon=[lon],lat=[lat])
            temp = self.__waternode(lon, lat, depth, url)
            nodes['lon'].extend(temp['lon'])
            nodes['lat'].extend(temp['lat'])
        else:
            nodes = dict(lon=[lon],lat=[lat])
            for i in url:
                temp = self.__waternode(nodes['lon'][-1], nodes['lat'][-1], depth, i)
                nodes['lat'].extend(temp['lat'])
                nodes['lon'].extend(temp['lon'])
        return nodes
    def __waternode(self, lon, lat, depth, url):
        '''
        start, end: indices of some period
        data: a dict that has 'u' and 'v'
        '''
        data = self.get_data(url)
        lonc, latc = data['lonc'][:], data['latc'][:]
        lonv, latv = data['lon'][:], data['lat'][:]
        h = data['h'][:]
        siglay = data['siglay'][:]
        if lon>90:
            lon, lat = dm2dd(lon, lat)
        nodes = dict(lon=[], lat=[])
        kf,distanceF = self.nearest_point_index(lon,lat,lonc,latc,num=1)
        kv,distanceV = self.nearest_point_index(lon,lat,lonv,latv,num=1)
        print 'kf', kf
        if h[kv] < 0:
            sys.exit('Sorry, your position is on land, please try another point')
        depth_total = siglay[:,kv]*h[kv]
        ###############layer###########################
        layer = np.argmin(abs(depth_total-depth))
        # for i in range(len(data['u'])):
        for i in range(self.hours):
            # u_t = np.array(data['u'])[i,layer,kf]
            # v_t = np.array(data['v'])[i,layer,kf]
            u_t = data['u'][i, layer, kf[0][0]]
            v_t = data['v'][i, layer, kf[0][0]]
            print 'u_t, v_t, i', u_t, v_t, i
            dx = 60*60*u_t
            dy = 60*60*v_t
            lon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            lat = lat + dy/111111
            nodes['lon'].append(lon)
            nodes['lat'].append(lat)
            kf, distanceF = self.nearest_point_index(lon, lat, lonc, latc,num=1)
            kv, distanceV = self.nearest_point_index(lon, lat, lonv, latv,num=1)
            # depth_total = siglay[:][kv]*h[kv]
            if distanceV>=.3:
                if i==start:
                    print 'Sorry, your start position is NOT in the model domain'
                    break
        return nodes
class water_drifter(water):
    def __init__(self, drifter_id):
        self.drifter_id = drifter_id
    def waternode(self, starttime=None, days=None):
        '''
        return drifter nodes
        if starttime is given, return nodes started from starttime
        if both starttime and days are given, return nodes of the specific time period
        '''
        nodes = {}
        temp = getdrift(self.drifter_id)
        nodes['lon'] = np.array(temp[1])
        nodes['lat'] = np.array(temp[0])
        nodes['time'] = np.array(temp[2])
        if bool(starttime):
            if bool(days):
                endtime = starttime + timedelta(days=days)
                i = self.__cmptime(starttime, nodes['time'])
                j = self.__cmptime(endtime, nodes['time'])
                nodes['lon'] = nodes['lon'][i:j+1]
                nodes['lat'] = nodes['lat'][i:j+1]
                nodes['time'] = nodes['time'][i:j+1]
            else:
                i = self.__cmptime(starttime, nodes['time'])
                nodes['lon'] = nodes['lon'][i:-1]
                nodes['lat'] = nodes['lat'][i:-1]
                nodes['time'] = nodes['time'][i:-1]
        return nodes
    def __cmptime(self, time, times):
        '''
        return indies of specific or nearest time in times.
        '''
        tdelta = []
        for t in times:
            tdelta.append(abs((time-t).total_seconds()))
        index = tdelta.index(min(tdelta))
        return index
class water_roms_rk4(water_roms):
    '''
    model roms using Runge Kutta
    '''
    def waternode(self, lon, lat, depth, url):
        '''
        get the nodes of specific time period
        lon, lat: start point
        url: get from get_url(starttime, endtime)
        depth: 0~35, the 36th is the bottom.
        '''
        self.startpoint = lon, lat
        if type(url) is str:
            nodes = self.__waternode(lon, lat, depth, url)
        else: # case where there are two urls, one for start and one for stop time
            nodes = dict(lon=[self.startpoint[0]],lat=[self.startpoint[1]])
            for i in url:
                temp = self.__waternode(nodes['lon'][-1], nodes['lat'][-1], depth, i)
                nodes['lon'].extend(temp['lon'][1:])
                nodes['lat'].extend(temp['lat'][1:])
        return nodes # dictionary of lat and lon
    def __waternode(self, lon, lat, depth, url):
        data = self.get_data(url)
        nodes = dict(lon=lon, lat=lat)
        mask = data['mask_rho'][:]
        lon_rho = data['lon_rho'][:]
        lat_rho = data['lat_rho'][:]
        # lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        # lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        index, nearestdistance = self.nearest_point_index(lon,lat,lons,lats)
        depth_layers = data['h'][index[0][0]][index[0][1]]*data['s_rho']
        layer = np.argmin(abs(depth_layers+depth))
        u = data['u'][:,layer]
        v = data['v'][:,layer]
        # lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        # lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        for i in range(0, len(data['u'][:])):
            # u_t = jata.shrink(u[i], mask[1:,1:].shape)
            # v_t = jata.shrink(v[i], mask[1:,1:].shape)
            u_t = u[i, :-2, :]
            v_t = v[i, :, :-2]
            lon, lat, u_p, v_p = self.RungeKutta4_lonlat(lon,lat,lons,lats,u_t,v_t)
            if not u_p:
                print 'point hit the land'
                break
            nodes['lon'] = np.append(nodes['lon'],lon)
            nodes['lat'] = np.append(nodes['lat'],lat)
        return nodes
    def polygonal_barycentric_coordinates(self,xp,yp,xv,yv):
        N=len(xv)   
        j=np.arange(N)
        ja=(j+1)%N
        jb=(j-1)%N
        Ajab=np.cross(np.array([xv[ja]-xv[j],yv[ja]-yv[j]]).T,
                      np.array([xv[jb]-xv[j],yv[jb]-yv[j]]).T)
        Aj=np.cross(np.array([xv[j]-xp,yv[j]-yp]).T,
                    np.array([xv[ja]-xp,yv[ja]-yp]).T)
        Aj=abs(Aj)
        Ajab=abs(Ajab)
        Aj=Aj/max(abs(Aj))
        Ajab=Ajab/max(abs(Ajab))    
        w=xv*0.
        j2=np.arange(N-2)
        for j in range(N):
            w[j]=Ajab[j]*Aj[(j2+j+1)%N].prod()
            # print Ajab[j],Aj[(j2+j+1)%N]
        w=w/w.sum()
        return w
    def VelInterp_lonlat(self,lonp,latp,lons,lats,u,v):
        '''
    # find the nearest vertex    
        kv,distance=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
     #   print kv,lonp,latp
    # list of triangles surrounding the vertex kv    
        kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
    # coordinates of the (dual mesh) polygon vertices: the centers of triangle faces
        lonv=Grid['lonc'][kfv];latv=Grid['latc'][kfv]
        w=polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
    # baricentric coordinates are invariant wrt coordinate transformation (xy - lonlat), check!    
    # interpolation within polygon, w - normalized weights: w.sum()=1.    
    # use precalculated Lame coefficients for the spherical coordinates
    # coslatc[kfv] at the polygon vertices
    # essentially interpolate u/cos(latitude)
    # this is needed for RungeKutta_lonlat: dlon = u/cos(lat)*tau, dlat = vi*tau
        cv=Grid['coslatc'][kfv]
     #   print cv    
        urci=(u[kfv]/cv*w).sum()
        vi=(v[kfv]*w).sum()
        return urci,vi
        '''
        index, distance = self.nearest_point_index(lonp,latp,lons,lats)
        lonv,latv = lons[index[0],index[1]], lats[index[0],index[1]]
        w = self.polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
        uf = (u[index[0],index[1]]/np.cos(lats[index[0],index[1]]*np.pi/180)*w).sum()
        vf = (v[index[0],index[1]]*w).sum()
        return uf, vf
    def RungeKutta4_lonlat(self,lon,lat,lons,lats,u,v):
        tau = 60*60/111111.
        lon1=lon*1.;          lat1=lat*1.;        urc1,v1=self.VelInterp_lonlat(lon1,lat1,lons,lats,u,v);  
        lon2=lon+0.5*tau*urc1;lat2=lat+0.5*tau*v1;urc2,v2=self.VelInterp_lonlat(lon2,lat2,lons,lats,u,v);
        lon3=lon+0.5*tau*urc2;lat3=lat+0.5*tau*v2;urc3,v3=self.VelInterp_lonlat(lon3,lat3,lons,lats,u,v);
        lon4=lon+    tau*urc3;lat4=lat+    tau*v3;urc4,v4=self.VelInterp_lonlat(lon4,lat4,lons,lats,u,v);
        lon=lon+tau/6.*(urc1+2.*urc2+2.*urc3+urc4);
        lat=lat+tau/6.*(v1+2.*v2+2.*v3+v4); 
        uinterplation=  (urc1+2.*urc2+2.*urc3+urc4)/6    
        vinterplation= (v1+2.*v2+2.*v3+v4)/6
       # print urc1,v1,urc2,v2,urc3,v3,urc4,v4
        return lon,lat,uinterplation,vinterplation

def min_data(*args):
    '''
    return the minimum of several lists
    '''
    data = []
    for i in range(len(args)):
        data.append(min(args[i]))
    return min(data)
def max_data(*args):
    '''
    return the maximum of several lists
    '''
    data = []
    for i in range(len(args)):
        data.append(max(args[i]))
    return max(data)
def angle_conversion(a):
    a = np.array(a)
    return a/180*np.pi
def dist(lon1, lat1, lon2, lat2):
    # calculate the distance of points
    R = 6371.004
    lon1, lat1 = angle_conversion(lon1), angle_conversion(lat1)
    lon2, lat2 = angle_conversion(lon2), angle_conversion(lat2)
    l = R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+
                    np.sin(lat1)*np.sin(lat2))
    return l
def draw_basemap(fig, ax, lonsize, latsize, interval_lon=0.5, interval_lat=0.5):
    ax = fig.sca(ax)
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
'''
modelname = 'ROMS'
if modelname is 'drifter':
    # starttime = datetime(year=2013, month=9, day=29, hour=11, minute=46)
    drifter_id = jata.input_with_default('drifter_id', 106420701)
    # dataloc = "/net/home3/ocn/jmanning/py/jc/web_track/drift_tcs_2013_1.dat"
    
    drifter = water_drifter(drifter_id)
    nodes = drifter.waternode()
    lonsize = min(nodes['lon'])-1, max(nodes['lon'])+1
    latsize = min(nodes['lat'])-1, max(nodes['lat'])+1
    fig = figure_with_basemap(lonsize, latsize)
    plt.plot(nodes['lon'], nodes['lat'],'ro-')
    plt.show()
elif modelname is 'ROMS':
    # startpoint = (-73, 38.0)  #point wanted to be forecast
    startpoint = (-70.40358, 41.494803)
    days = 2                  #forecast 3 days later, and show [days-3] days before
    isub = 3                  #interval of arrow of water speed
    scale = 0.03              #
    tidx = -1                 #layer. -1 is the last one.
    starttime =datetime(year=2014,month=2,day=25)
    endtime = starttime + timedelta(days=days)
    
    water_roms = water_roms()
    url = water_roms.get_url(starttime, endtime)
    nodes = water_roms.waternode(startpoint[0], startpoint[1], url)
    # lonc = data['lon_rho'][1:-1, 1:-1]
    # latc = data['lat_rho'][1:-1, 1:-1]
    # u = data['u'][:, -1][tidx,:,:]
    # v = data['v'][:, -1][tidx,:,:]
    # u = jata.shrink(u, data['mask_rho'][1:-1, 1:-1].shape)
    # v = jata.shrink(v, data['mask_rho'][1:-1, 1:-1].shape)
    lonsize = min(nodes['lon'])-1, max(nodes['lon'])+1
    latsize = min(nodes['lat'])-1, max(nodes['lat'])+1
    fig = figure_with_basemap(lonsize, latsize)
    # fig.ax.quiver(lonc[::isub,::isub], latc[::isub,::isub],
    #               u[::isub,::isub], v[::isub,::isub],
    #               scale=1.0/scale, pivot='middle',
    #               zorder=1e35, width=0.003,color='blue')
    plt.plot(nodes['lon'], nodes['lat'], 'ro-')
    plt.show()
elif modelname is 'FVCOM':
    model = 'massbay'
    days = 2
    #when you choose '30yr' model, please keep
    #starttime before 2010-12-31 after 1978.
    starttime = '2014-2-25 13:40:00'
    lon = -70.718466
    lat = 40.844644
    # lon = float(jata.input_with_default('lon', 7031.8486))
    # lat = float(jata.input_with_default('lat', 3934.4644))
    # starttime = jata.input_with_default('TIME','2014-02-07 13:40:00')
    starttime = datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
    depth = -3
    
    water_fvcom = water_fvcom(model)
    # dataloc, index = water_fvcom.get_interval(starttime)
    # data = water_fvcom.get_data()
    url  = water_fvcom.get_url(starttime, days)
    nodes = water_fvcom.waternode(lon, lat, depth, url)
    lonsize = min(nodes['lon'])-1, max(nodes['lon'])+1
    latsize = min(nodes['lat'])-1, max(nodes['lat'])+1
    fig = figure_with_basemap(lonsize, latsize)
    fig.ax.plot(nodes['lon'], nodes['lat'], 'ro-')
    plt.show()
'''
########################main code###########################
drifter_id = 118410701
days = 3
depth = -1
starttime = '2011-08-02 00:00'

starttime = datetime.strptime(starttime, '%Y-%m-%d %H:%M')
drifter_id = jata.input_with_default('drifter_id', drifter_id)
drifter = water_drifter(drifter_id)
if starttime:
    if days:
        nodes_drifter = drifter.waternode(starttime,days)
    else:
        nodes_drifter = drifter.waternode(starttime)
else:
    nodes_drifter = drifter.waternode()

lon, lat = nodes_drifter['lon'][0], nodes_drifter['lat'][0]
starttime = nodes_drifter['time'][0]
endtime = nodes_drifter['time'][-1]

water_fvcom =  water_fvcom()
url_fvcom = water_fvcom.get_url(starttime, endtime)
nodes_fvcom = water_fvcom.waternode(lon,lat,depth,url_fvcom)
water_roms = water_roms()
url_roms = water_roms.get_url(starttime, endtime)
nodes_roms = water_roms.waternode(lon, lat, depth, url_roms)
'''
water_roms_rk4 = water_roms_rk4()
url_roms_rk4 = water_roms_rk4.get_url(starttime, endtime)
nodes_roms_rk4 = water_roms_rk4.waternode(lon, lat, depth, url_roms_rk4)
'''
lonsize = [min_data(nodes_drifter['lon'],nodes_roms['lon'])-0.5,
           max_data(nodes_drifter['lon'],nodes_roms['lon'])+0.5]
latsize = [min_data(nodes_drifter['lat'],nodes_roms['lat'])-0.5,
           max_data(nodes_drifter['lat'],nodes_roms['lat'])+0.5]
fig = plt.figure()
ax = fig.add_subplot(111)
draw_basemap(fig, ax, lonsize, latsize)
ax.plot(nodes_drifter['lon'],nodes_drifter['lat'],'ro-',label='drifter')
# ax.plot(nodes_roms_rk4['lon'],nodes_roms_rk4['lat'],'bo-',label='roms_rk4')
ax.plot(nodes_fvcom['lon'],nodes_fvcom['lat'],'yo-',label='fvcom')
ax.plot(nodes_roms['lon'],nodes_roms['lat'], 'go-', label='roms')
plt.annotate('Startpoint', xy=(lon, lat), arrowprops=dict(arrowstyle='simple'))
plt.title('ID: {0} {1} {2} days'.format(drifter_id, starttime, days))
plt.legend(loc='lower right')
# figname = 'track_cmp-{0}-{1}-{2}.png'.format(drifter_id, starttime, days)
# plt.savefig(figname, dpi=200)
plt.show()

'''calculate the distance between model and observation. not good, because drifter loses some points.
f = min(len(nodes_drifter['lon']), len(nodes_fvcom['lon']))
dist_fvcom = dist(nodes_fvcom['lon'][:f],nodes_fvcom['lat'][:f],
                  nodes_drifter['lon'][:r_rk4],nodes_drifter['lat'][:r_rk4])
r = min(len(nodes_drifter['lon']), len(nodes_roms['lon']))
dist_roms = dist(nodes_roms['lon'][0:r],nodes_roms['lat'][0:r],
                 nodes_drifter['lon'][0:r],nodes_drifter['lat'][0:r])
f = min(len(nodes_drifter['lon']), len(nodes_fvcom['lon']))
dist_fvcom = dist(nodes_fvcom['lon'][:f],nodes_fvcom['lat'][:f],
                  nodes_drifter['lon'][:f],nodes_drifter['lat'][:f])
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.plot(dist_roms, 'r-', label='roms')
# plt.plot(dist_roms_rk4, 'b-', label='roms_rk4')
# plt.plot(dist_fvcom, 'y-', label='fvcom')
plt.legend(loc='lower right')
plt.title('Distance of drifter data and model data')
plt.show()
'''
