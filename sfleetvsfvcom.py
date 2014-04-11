import matplotlib.pyplot as plt
import numpy as np
from numpy import float64
from datetime import datetime
from datetime import timedelta
import pandas as pd
import sys
import netCDF4
from matplotlib import path
from dateutil import rrule
# import our local modules
sys.path.append('../modules')
import jmath,jata
from conversions import dm2dd,f2c
from utilities import my_x_axis_format

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
class water_fvcom(water):
    def __init__(self):
        self.modelname = '30yr'
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
                index1 = 26340+35112*(yearnum/4)+8772*(yearnum%4)+1+self.hours
                index2 = index1 + self.hours
                furl = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?h[0:1:48450],lat[0:1:48450],latc[0:1:90414],lon[0:1:48450],lonc[0:1:90414],u[{0}:1:{1}][0:1:44][0:1:90414],v[{0}:1:{1}][0:1:44][0:1:90414],siglay,temp[{0}:1:{1}][0:1:44][0:1:48450],time[{0}:1:{1}]'
                url.append(furl.format(index1, index2)) 
            elif time1 <= endtime < time2: # endtime is in GOM3_v11
                url.extend(self.__temp(starttime,endtime,time1,time2))
            elif time2 <= endtime < time3:  # endtime is in GOM3_v12
                url.extend(self.__temp(starttime,endtime,time2,time3))
            elif time3 <= endtime < time4:
                url.extend(self.__temp(starttime,endtime,time3,time4))
        elif self.modelname is "GOM3":
            url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?lon[0:1:51215],lat[0:1:51215],lonc[0:1:95721],latc[0:1:95721],siglay[0:1:39][0:1:51215],h[0:1:51215],u[{0}:1:{1}][0:1:39][0:1:95721],v[{0}:1:{1}][0:1:39][0:1:95721],temp[{0}:1:{1}][0:1:44][0:1:48450],time[{0}:1:{1}]'
            period = starttime-\
                     (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
            index1 = period.total_seconds()/60/60
            index2 = index1 + self.hours
            url = url.format(index1, index2)
        elif self.modelname is "massbay":
            url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?lon[0:1:98431],lat[0:1:98431],lonc[0:1:165094],latc[0:1:165094],siglay[0:1:9][0:1:98431],h[0:1:98431],u[{0}:1:{1}][0:1:9][0:1:165094],v[{0}:1:{1}][0:1:9][0:1:165094],temp[{0}:1:{1}][0:1:44][0:1:48450],time[{0}:1:{1}]'
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
            url.extend(self.get_url(time1,endtime))
        return url
    def __url(self, year, month, start_daytime, end_daytime):
        '''
        start_daytime,end_daytime: [day,hour]
        '''
        url_v11 = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM3_{0}/gom3v11_{0}{1}.nc?lon[0:1:48727],lat[0:1:48727],lonc[0:1:90997],latc[0:1:90997],h[0:1:48727],u[{2}:1:{3}][0:1:39][0:1:90997],v[{2}:1:{3}][0:1:39][0:1:90997],siglay[0:1:39][0:1:48727],temp[{0}:1:{1}][0:1:44][0:1:48450],time[{0}:1:{1}]'
        url_v12 = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM3_{0}/gom3v12_{0}{1}.nc?lon[0:1:48859],lat[0:1:48859],lonc[0:1:91257],latc[0:1:91257],h[0:1:48859],u[{2}:1:{3}][0:1:39][0:1:91257],v[{2}:1:{3}][0:1:39][0:1:91257],siglay[0:1:39][0:1:48859],temp[{0}:1:{1}][0:1:44][0:1:48450],time[{0}:1:{1}]'
        url_v13 = 'http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM3_{0}/gom3v13_{0}{1}.nc?lon[0:1:51215],lat[0:1:51215],lonc[0:1:95721],latc[0:1:95721],h[0:1:51215],u[{2}:1:{3}][0:1:39][0:1:95721],v[{2}:1:{3}][0:1:39][0:1:95721],siglay[0:1:39][0:1:51215],temp[{0}:1:{1}][0:1:44][0:1:48450],time[{0}:1:{1}]'
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
                                     'u','v','siglay','h','time','temp')
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
        if h[kv] < 0:
            sys.exit('Sorry, your position is on land, please try another point')
        depth_total = siglay[:,kv]*h[kv]
        ###############layer###########################
        layer = np.argmin(abs(depth_total-depth))
        # for i in range(len(data['u'])):
        for i in range(self.hours):
            u_t = data['u'][i, layer, kf[0][0]]
            v_t = data['v'][i, layer, kf[0][0]]
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
        modtso = pd.DataFrame(nodes, )
        return nodes
    def watertemp(self, lon, lat, depth, url):
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
        if h[kv] < 0:
            sys.exit('Sorry, your position is on land, please try another point')
        depth_total = siglay[:,kv]*h[kv]
        ###############layer###########################
        layer = np.argmin(abs(depth_total-depth))
        print 'layer, kf', layer, kf[0][0]
        # for i in range(len(data['time'][:])):
        temp = data['temp'][:, layer, kf[0][0]]
        print 'l'
        m = pd.DataFrame(temp, index=data['time'][:])
        print 'a'
        return m

def inconvexpolygon(xp,yp,xv,yv):
    """
check if point is inside a convex polygon

    i=inconvexpolygon(xp,yp,xv,yv)
    
    xp,yp - arrays of points to be tested
    xv,yv - vertices of the convex polygon

    i - boolean, True if xp,yp inside the polygon, False otherwise
    
    """    
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N # next vertex in the sequence 
#    jb=(j-1)%N # previous vertex in the sequence
    
    NP=len(xp)
    i=np.zeros(NP,dtype=bool)
    for k in range(NP):
        # area of triangle p,j,j+1
        print j
        print xv[j]
        print xp[k]
        Aj=np.cross(np.array([xv[j]-xp[k],yv[j]-yp[k]]).T,np.array([xv[ja]-xp[k],yv[ja]-yp[k]]).T) 
    # if a point is inside the convect polygon all these Areas should be positive 
    # (assuming the area of polygon is positive, counterclockwise contour)
        Aj /= Aj.sum()
    # Now there should be no negative Aj
    # unless the point is outside the triangular mesh
        i[k]=(Aj>0.).all()
        
    return i
def inmodel_fvcom(xp, yp):
    fvcomleft_lat, fvcomleft_lon = 38.3263045, -77.0485845
    fvcomtop_lat, fvcomtop_lon = 48.081465, -61.290788
    fvcomright_lat, fvcomright_lon = 42.274921, -56.8508
    fvcombottom_lat, fvcombottom_lon = 35.283871, -82.955833
    fvcom_xv = np.array([fvcomtop_lon, fvcomright_lon, fvcombottom_lon, fvcomleft_lon])
    fvcom_yv = np.array([fvcomtop_lat, fvcomright_lat, fvcombottom_lat, fvcomleft_lat])
    i = inconvexpolygon(xp, yp, fvcom_xv, fvcom_yv)
    return i

# url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?lat[0:1:48450],lon[0:1:48450]'
# data = netCDF4.Dataset(url)

fn='binned_td_test.csv'
direct='/net/data5/jmanning/hoey/' 
outputdir='/net/data5/jmanning/modvsobs/sfleet/'
# read in the study fleet data
def parse(datet):
   #print datet[0:10],datet[11:13],datet[14:16]
    dtt=datetime.strptime(datet,'%Y-%m-%d:%H:%M')
    return dtt
obstso=pd.read_csv(direct+fn,sep=',',skiprows=0,parse_dates={'datet':[2]},index_col='datet',date_parser=parse,names=['LATITUDE','LONGITUDE','ROUND_DATE_TIME','OBSERVATIONS_TEMP','MEAN_TEMP','MIN_TEMP','MAX_TEMP','STD_DEV_TEMP','OBSERVATIONS_DEPTH','MEAN_DEPTH','MIN_DEPTH','MAX_DEPTH','STD_DEV_DEPTH','nan'])
#convert ddmm.m to dd.ddd
plt.figure(figsize=(16,10))
o=[]
m=[]
print '1'
for k in range(len(obstso)): # 
        [la,lo]=dm2dd(obstso['LATITUDE'][k],obstso['LONGITUDE'][k])
        if not inmodel_fvcom([lo], [la]):
           print 'point not in fvcom domain'
           continue
        else:
            print '2'
            fvcomobj = water_fvcom()
            print '3'
            # modtso = pd.DataFrame()
            st=obstso.index[k]
            print '4'
            et=st+timedelta(hours=1)
            print '5'
            url_fvcom = fvcomobj.get_url(st,et )
            print '6'
            modtso= fvcomobj.watertemp(lo, la,-1*obstso['MEAN_DEPTH'][k], url_fvcom[0])
            print 'modtso.values[0]', modtso.values[0]
            print 'run here'
            o.append(obstso['MEAN_TEMP'][k])
            m.append(modtso.values[0])
            plt.plot(modtso.values[0],obstso['MEAN_TEMP'][k],'*', markersize=30)
plt.xlabel('MODEL (degC)')
plt.ylabel('OBSERVATIONS (degC)')
plt.title('Study Fleet vs FVCOM ')
plt.show()
