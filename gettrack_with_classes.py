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
import sys
from getdata import getdrift
import calendar

class figure_with_basemap(mpl.figure.Figure):
    def __init__(self,lonsize,latsize,axes_num=1,interval_lon=0.5,interval_lat=0.5):
        '''
        draw the Basemap, set the axes num in the figure
        '''
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
    def __init__(self, startpoint):
        '''
        get startpoint of water, and the location of datafile.
        startpoint = [25,45]
        '''
        self.startpoint = startpoint
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
    def nearest_point_index(self, lon, lat, lons, lats, length=(1, 10)):
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
        ''' get several nearest points
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
        mindist = np.argmin(dist)
        indx = [i[mindist] for i in index]
        return indx, dist[mindist]
    def waternode(self, timeperiod, data):
        pass
class water_roms(water):
    '''
    use two urls:
        ####old, dayly####
        (2009.10.11, 2013.05.19):version1(old) 2009-2013
        (2013.05.19, present): version2(new) 2013-present
        ####new, hourly####
        (2006.01.01.01:00, present)
    '''
    def __init__(self):
        pass
        # self.startpoint = lon, lat
        # self.dataloc = self.get_url(starttime)
    def get_url(self, starttime, endtime):
        '''
        get url according to starttime and endtime, maybe string or maybe lists.
        '''
        '''
        self.starttime = starttime
        self.days = int((endtime-starttime).total_seconds()/60/60/24)+1 # get total days
        time1 = datetime(year=2009,month=10,day=11) # time of url1 that starts from
        time2 = datetime(year=2013,month=5,day=19)  # time of url2 that starts from
        url1 = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2009_da/avg?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
        url2 = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2013_da/avg_Best/ESPRESSO_Real-Time_v2_Averages_Best_Available_best.ncd?mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129],lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129]'
        if endtime >= time2:
            if starttime >=time2:
                index1 = (starttime - time2).days
                index2 = index1 + self.days
                url = url2.format(index1, index2)
            elif time1 <= starttime < time2:
                url = []
                index1 = (starttime - time1).days
                url.append(url1.format(index1, 1316))
                url.append(url2.format(0, self.days))
        elif time1 <= endtime < time2:
            index1 = (starttime-time1).days
            index2 = index1 + self.days
            url = url1.format(index1, index2)
        return url
        '''
        self.starttime = starttime
        self.hours = int((endtime-starttime).total_seconds()/60/60) # get total hours
        time_r = datetime(year=2006,month=1,day=9,hour=1,minute=0)
        index1 = (starttime - time_r).total_seconds()/60/60
        index2 = index1 + self.hours
        print 'time', index1, index2
        url = 'http://tds.marine.rutgers.edu:8080/thredds/dodsC/roms/espresso/2006_da/his?lon_rho[0:1:81][0:1:129],lat_rho[0:1:81][0:1:129],mask_rho[0:1:81][0:1:129],u[{0}:1:{1}][0:1:35][0:1:81][0:1:128],v[{0}:1:{1}][0:1:35][0:1:80][0:1:129]'
        url = url.format(index1, index2)
        return url
    def get_data(self, url):
        '''
        return the data needed.
        url is from water_roms.get_url(starttime, endtime)
        '''
        data = jata.get_nc_data(url, 'lon_rho', 'lat_rho', 'mask_rho','u', 'v')
        return data
    def waternode(self, lon, lat, url):
        '''
        get the nodes of specific time period
        lon, lat: start point
        url: get from get_url(starttime, endtime)
        '''
        self.startpoint = lon, lat
        if type(url) is str:
            nodes = self.__waternode(lon, lat, url)
        else:
            nodes = dict(lon=[self.startpoint[0]],lat=[self.startpoint[1]])
            for i in url:
                temp = self.__waternode(nodes['lon'][-1], nodes['lat'][-1], i)
                nodes['lon'].extend(temp['lon'][1:])
                nodes['lat'].extend(temp['lat'][1:])
        return nodes
    def __waternode(self, lon, lat, url):
        '''
        return points
        '''
        self.data = self.get_data(url)
        nodes = dict(lon=lon, lat=lat)
        mask = self.data['mask_rho'][:]
        lon_rho = self.data['lon_rho'][:]
        lat_rho = self.data['lat_rho'][:]
        u = self.data['u'][:,-1]
        v = self.data['v'][:,-1]
        lons = jata.shrink(lon_rho, mask[1:,1:].shape)
        lats = jata.shrink(lat_rho, mask[1:,1:].shape)
        print 'lons', len(lons),len(lons[0])
        for i in range(0, self.hours):
            print 'roms',i
            u_t = jata.shrink(u[i], mask[1:,1:].shape)
            v_t = jata.shrink(v[i], mask[1:,1:].shape)
            index, nearestdistance = self.nearest_point_index(lon,lat,lons,lats)
            print 'index', index
            print 'u_t', len(u_t), len(u_t[0])
            u_p = u_t[index[0]][index[1]]
            v_p = v_t[index[0]][index[1]]
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
            nodes['lon'] = np.append(nodes['lon'],lon)
            nodes['lat'] = np.append(nodes['lat'],lat)
        return nodes
class water_fvcom(water):
    def __init__(self, modelname):
        '''
        starttime: datetime.datetime()
        '''
        self.modelname = modelname
        # if self.modelname is '30yr':
        #     self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
        #                    ','.join(datakeys)
        #     yearnum = starttime.year-1981
        #     standardtime = datetime.strptime(str(starttime.year)+'-01-01 00:00:00',
        #                                      '%Y-%m-%d %H:%M:%S')
        #     index1 = 26340+35112*(yearnum/4)+8772*(yearnum%4)+1+\
        #               24*(starttime-standardtime).days
        #     index2 = index1+24*self.days
        # elif self.modelname is 'GOM3':
        #     self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?'+\
        #                    ','.join(datakeys)
        #     period = (starttime+timedelta(days=self.days))-\
        #               (datetime.now()-timedelta(days=3))
        #     index1 = (period.seconds)/60/60
        #     index2 = index1 + 24*(self.days)
        # elif self.modelname is 'massbay':
        #     self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?'+\
        #                    ','.join(datakeys)
        #     period = (starttime+timedelta(days=self.days))-\
        #              (datetime.now()-timedelta(days=3))
        #     index1 = (period.seconds)/60/60
        #     index2 = index1+24*self.days
        # else:
        #     raise Exception('Please use right model')
        # self.index = [index1, index2]
    def get_url(self, starttime, endtime):
        '''
        get different url according to starttime and endtime.
        urls are monthly.
        '''
        self.hours = int((endtime-starttime).total_seconds()/60/60)
        if self.modelname is "30yr":
            url = []
            # endtime = starttime + timedelta(days=days)
            time1 = datetime(year=2011,month=1,day=1)      #all these datetime are made based on the model.
            time2 = datetime(year=2011,month=11,day=11)      #The model use different version data of different period.
            time3 = datetime(year=2013,month=05,day=9)
            time4 = datetime(year=2013,month=12,day=1)
            # endtime = starttime + timedelta(days=days)
            if endtime < time1:
                yearnum = starttime.year-1981
                standardtime = datetime.strptime(str(starttime.year)+'-01-01 00:00:00',
                                                 '%Y-%m-%d %H:%M:%S')
                index1 = 26340+35112*(yearnum/4)+8772*(yearnum%4)+1+self.hours
                index2 = index1 + self.hours
                furl = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?h[0:1:48450],lat[0:1:48450],latc[0:1:90414],lon[0:1:48450],lonc[0:1:90414],u[{0}:1:{1}][0:1:44][0:1:90414],v[{0}:1:{1}][0:1:44][0:1:90414],siglay'
                url.append(furl.format(index1, index2)) 
            elif time1 <= endtime < time2: # endtime is in GOM3_v11
                url.extend(self.__temp(starttime,endtime,time1,time2))
                # if starttime > time1:
                #     if starttime.month == endtime.month:
                #         url.append(__url(11,starttime.month,
                #                                [starttime.day, starttime.hour],
                #                                [endtime.day, endtime.hour]))
                #     else:
                #         for i in range(starttime.month, endtime.month+1):
                #             if i == starttime.month:
                #                 url.append(__url(11,i,
                #                                        [starttime.day,starttime.hour],
                #                                        [calendar.monthrange(2011,i)[1],0]))
                #             elif starttime.month < i < endtime.month:
                #                 url.append(__url(11,i,[0,0],
                #                                        [calendar.monthrange(2011,i)[1],0]))
                #             elif i == endtime.month:
                #                 url.append(__url(11,i,[0,0],
                #                                        [endtime.day,endtime.hour]))
                # elif starttime <= time1: # start time  is from 1978 to 2010
                #     url.extend(get_url(starttime, time1))
                #     url.extend(get_url(time1+timedelta(days=1), endtime))
            elif time2 <= endtime < time3:  # endtime is in GOM3_v12
                url.extend(self.__temp(starttime,endtime,time2,time3))
                # if starttime > time2:    #start time is from 2011.11.10 as v12
                #     if starttime.month == endtime.month:
                #         url.append(__url(starttime.year,starttime.month,
                #                                [starttime.day,starttime.hour],
                #                                [endtime.day,endtime.hour]))
                #     else:
                #         if starttime.year == endtime.year:
                #             y = starttime.year
                #             for i in range(starttime.month, endtime.month+1):
                #                 if i == starttime.month:
                #                     url.append(__url(y,i,
                #                                            [starttime.month, starttime.hour],
                #                                            [calender.monthrange(y,i)[1],0]))
                #                 elif starttime.month < i < endtime.month:
                #                     url.append(__url(y,i,[0,0],
                #                                            [calendar.monthrange(y,i)[1],0]))
                #                 elif i == endtime.month:
                #                     url.append(__url(y,i,[0,0],
                #                                            [endtime.day,endtime.hour]))
                #         else:
                #             for i in range(starttime.year, endtime.year+1):
                #                 if i == starttime.year:
                #                     url.extend(get_url(starttime,
                #                                        datetime(year=i,
                #                                                 month=12,day=31)))
                #                 elif i == endtime.year:
                #                     url.extend(get_url(datetime(year=i,month=1,day=1),
                #                                        endtime))
                #                 else:
                #                     url.extend(get_url(datetime(year=i,month=1,day=1),
                #                                        datetime(year=i,month=12,day=31)))
                     
                # else:
                #     url.extend(get_url(starttime,time2))
                #     url.extend(get_url(datetime(year=2011,month=11,day=11),endtime))
            elif time3 <= endtime < time4:
                url.extend(self.__temp(starttime,endtime,time3,time4))
                # if starttime > time3:
                #     if starttime.month == endtime.month:
                #         url.append(__url(starttime.year,starttime.month,
                #                                [starttime.day,starttime.hour],
                #                                [endtime.day,endtime.hour]))
                #     else:
                #         y = starttime.year
                #         for i in range(starttime.month, endtime.month+1):
                #             if i == starttime.month:
                #                 url.append(__url(y,i,
                #                                        [starttime.month,starttime.hour],
                #                                        [calender.monthrange(y,i)[1],0]))
                #             elif
        elif self.modelname is "GOM3":
            url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?lon[0:1:51215],lat[0:1:51215],lonc[0:1:95721],latc[0:1:95721],siglay[0:1:39][0:1:51215],h[0:1:51215],u[{0}:1:{1}][0:1:39][0:1:95721],v[{0}:1:{1}][0:1:39][0:1:95721]'
            period = starttime-\
                     (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
            index1 = period.total_seconds()/60/60
            index2 = index1 + self.hours
            url = url.format(index1, index2)
        elif self.modelname is "massbay":
            url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?lon[0:1:98431],lat[0:1:98431],lonc[0:1:165094],latc[0:1:165094],siglay[0:1:9][0:1:98431],h[0:1:98431],u[{0}:1:{1}][0:1:9][0:1:165094],v[{0}:1:{1}][0:1:9][0:1:165094]'
            period = starttime-\
                     (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
            index1 = period.total_seconds()/60/60
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
            # url.extend(self.get_url(datetime(year=2011,month=11,day=10),endtime))
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
    # def get_interval(self, starttime):
    #     '''
    #     starttime: datetime.datetime()
    #     '''
    #     days_int = self.days
    #     days_datetime = timedelta(days=self.days)
    #     datakeys = ('u', 'v', 'lon', 'lat', 'lonc', 'latc', 'siglay', 'h')
    #     if self.modelname is '30yr':
    #         self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
    #                        ','.join(datakeys)
    #         yearnum = starttime.year-1981
    #         standardtime = datetime.strptime(str(starttime.year)+'-01-01 00:00:00',
    #                                          '%Y-%m-%d %H:%M:%S')
    #         index1 = 26340+35112*(yearnum/4)+8772*(yearnum%4)+1+\
    #                   24*(starttime-standardtime).days
    #         index2 = index1+24*days_int
    #     elif self.modelname is 'GOM3':
    #         self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?'+\
    #                        ','.join(datakeys)
    #         period = (starttime+days_datetime)-(datetime.now()-timedelta(days=3))
    #         index1 = (period.seconds)/60/60
    #         index2 = index1 + 24*(days_int)
    #     elif self.modelname is 'massbay':
    #         self.dataloc = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?'+\
    #                        ','.join(datakeys)
    #         period = (starttime+days_datetime)-(datetime.now()-timedelta(days=3))
    #         index1 = (period.seconds)/60/60
    #         index2 = index1+24*days_int
    #     else:
    #         raise Exception('Please use right model')
    #     return self.dataloc, [index1, index2]
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
        kf,distanceF = self.nearest_point_index(lon,lat,lonc,latc)
        kv,distanceV = self.nearest_point_index(lon,lat,lonv,latv)
        if h[kv] < 0:
            sys.exit('Sorry, your position is on land, please try another point')
        depth_total = siglay[:,kv]*h[kv]
        ###############layer###########################
        layer = np.argmin(abs(depth_total-depth))
        # for i in range(len(data['u'])):
        for i in range(self.hours):
            # u_t = np.array(data['u'])[i,layer,kf]
            # v_t = np.array(data['v'])[i,layer,kf]
            u_t = data['u'][i][layer][kf[0]]
            v_t = data['v'][i][layer][kf[0]]
            dx = 60*60*u_t
            dy = 60*60*v_t
            lon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            lat = lat + dy/111111
            nodes['lon'].append(lon)
            nodes['lat'].append(lat)
            kf, distanceF = self.nearest_point_index(lon, lat, lonc, latc)
            kv, distanceV = self.nearest_point_index(lon, lat, lonv, latv)
            # depth_total = siglay[:][kv]*h[kv]
            if distanceV>=.3:
                if i==start:
                    print 'Sorry, your start position is NOT in the model domain'
                    break
        return nodes
class water_drifter(water):
    def __init__(self, drifter_id):
        # self.dataloc = "/net/home3/ocn/jmanning/py/jc/web_track/drift_tcs_2013_1.dat"
        self.drifter_id = drifter_id
        # self.starttime = starttime
    def waternode(self, starttime=None, days=None):
        '''
        return drifter nodes
        if starttime is given, return nodes started from starttime
        if both starttime and days are given, return nodes of the specific time period
        '''
        # self.drifter_id = jata.input_with_default('drifter ID', 139420691)
        # self.starttime = datetime(year=2013, month=9, day=29, hour=11,minute=46)
        # nodes = jata.data_extracted(self.dataloc, self.drifter_id, self.starttime)
        nodes = {}
        temp = getdrift(self.drifter_id)
        nodes['lon'] = temp[1]
        nodes['lat'] = temp[0]
        nodes['time'] = temp[2]
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

#######################################110410712,117400701 
# drifter_id = jata.input_with_default('drifter_id', )
drifter_id = jata.input_with_default('drifter_id', 106410712)
days = 3
model = '30yr'
# starttime = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
# starttime = datetime(year=2013,month=9,day=22,hour=15,minute=47)

# starttime = '2011-10-10 15:47'           #if used, make sure it's in drifter period
starttime = '2010-07-25 00:00'
starttime = datetime.strptime(starttime, '%Y-%m-%d %H:%M')

# starttime = None
depth = -1

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

water_fvcom =  water_fvcom(model)
url_fvcom = water_fvcom.get_url(starttime, endtime)
nodes_fvcom = water_fvcom.waternode(lon,lat,depth,url_fvcom)

water_roms = water_roms()
url_roms = water_roms.get_url(starttime, endtime)
nodes_roms = water_roms.waternode(lon, lat, url_roms)
print 'nodes_roms', nodes_roms
print 'nodes_fvcom', nodes_fvcom

lonsize = [min_data(nodes_drifter['lon'],nodes_fvcom['lon'],nodes_roms['lon'])-1,
           max_data(nodes_drifter['lon'],nodes_fvcom['lon'],nodes_roms['lon'])+1]
latsize = [min_data(nodes_drifter['lat'],nodes_fvcom['lat'],nodes_roms['lat'])-1,
           max_data(nodes_drifter['lat'],nodes_fvcom['lat'],nodes_roms['lat'])+1]

fig = figure_with_basemap(lonsize, latsize)
fig.ax.plot(nodes_drifter['lon'],nodes_drifter['lat'],'ro-',label='drifter')
fig.ax.plot(nodes_roms['lon'],nodes_roms['lat'],'bo-',label='roms')
fig.ax.plot(nodes_fvcom['lon'],nodes_fvcom['lat'],'yo-',label='fvcom')
plt.annotate('Startpoint', xy=(lon, lat), arrowprops=dict(arrowstyle='simple'))
plt.title('ID: {0} {1} {2} days'.format(drifter_id, starttime, days))
plt.legend()
plt.show()
