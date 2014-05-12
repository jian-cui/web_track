"""
Created on Mon Jun 17 10:15:52 2013
This toutine reads a control file called ctrl_trackzoomin.csv
@author: jmanning
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
#import pylab
from datetime import datetime
from pydap.client import open_url
from datetime import timedelta
from conversions import dm2dd
import sys
import netCDF4
from matplotlib import path
#class figure_map(figure):

#la=4224.7 # this can be in decimal degrees instead of deg-minutesif it is easier to input
#lo=7005.7876
#urlname = raw_input('please input model name(massbay or 30yr): ')
#depth = int(raw_input('Please input the depth(negtive number): '))
#TIME = raw_input('Please input starttime(2013-10-18 00:00:00): ')
#numdays = int(raw_input('Please input numday(positive number): '))
#def isNum(value):
#    try:
#        float(value)
#    except(ValueError):
#        print("Please input a number")
#la = raw_input('Please input latitude(default 4150.1086): ')
#if la == '':
#    la = 4150.1086
#else:
#    isNum(la)
#    la = float(la)
#lo = raw_input('Please input longitude(default 7005.7876): ')
#if lo == '':
#    lo == 7005.7876
#else:
#    isNum(lo)
#    lo = float()
def bbox2ij(lons, lats, bbox):
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
def nearest_point_index(lon, lat, lons, lats, length=(1, 1),num=4):
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
    index = bbox2ij(lons, lats, bbox)
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
def input_with_default(data, v_default):
    '''
    return a str
    '''
    l = (data, str(v_default))
    data_input = raw_input('Please input %s(default %s): ' % l)
    if data_input == '':
        data_input = l[1]
    else:
        data_input = data_input
    return data_input

#la = float(input_with_default('lat', 4015.497))
#lo = float(input_with_default('lon', 6901.6878))
#############get the index of lat and lon???
def nearlonlat(lon,lat,lonp,latp):
    cp=np.cos(latp*np.pi/180.)
    # approximation for small distance
    dx=(lon-lonp)*cp
    dy=lat-latp
    dist2=dx*dx+dy*dy
    #dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
    min_dist=np.sqrt(dist2[i])
    return i,min_dist

#def get_url(urlname, stime, new_numdays = None):
#    '''
#    get the url of different model.
#    if urlname is GOM3 or massbay, new_numdays is necessary.
#    '''
#    if urlname=='30yr':
##        stime=datetime.strptime(TIME, "%Y-%m-%d %H:%M:%S")
#        timesnum=stime.year-1981
#        standardtime=datetime.strptime(str(stime.year)+'-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
#        timedeltaprocess=(stime-standardtime).days
#        startrecord=26340+35112*(timesnum/4)+8772*(timesnum%4)+1+timedeltaprocess*24
#        endrecord=startrecord+24*numdays
#        url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+
#            'lon,lat,latc,lonc,siglay,h,Times['+str(startrecord)+':1:'+str(startrecord)+']'
#    elif urlname == 'GOM3' and new_numdays:
#        timeperiod=(starttime+new_numdays)-(datetime.now()-timedelta(days=3))
#        startrecord=(timeperiod.seconds)/60/60
#        endrecord=startrecord+24*(new_numdays.days)
#        url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?"+
#            'lon,lat,latc,lonc,siglay,h,Times['+str(startrecord)+':1:'+str(startrecord)+']'
#    elif urlname == 'massbay' and new_numdays:
#        timeperiod=(starttime+new_numdays)-(datetime.now()-timedelta(days=3))
#        startrecord=(timeperiod.seconds)/60/60
#        endrecord=startrecord+24*(new_numdays.days)
#        url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?"+
#            'lon,lat,latc,lonc,siglay,h,Times['+str(startrecord)+':1:'+str(startrecord)+']'
#    else:
#        raise Exception('You need to input new_numdays if model name is massbay or GOM3)
#    return url, startrecord, endrecord

def get_indices(modelname, starttime, time_interval=None):
    '''
    Return a period of indices of certain model started from 'starttime'.
    time_interval is neccessary when use massbay or GOM3

    modelname: string
    starttime: datetime
    time_interval: timedelta (neccessary when use massbay or GOM3)
    '''
    if modelname == '30yr':
        timesnum = starttime.year-1981
        standardtime = datetime.strptime(str(starttime.year) + '-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
        timedeltaprocess = (starttime-standardtime).days
        startrecord = 26340+35112*(timesnum/4)+8772*(timesnum%4)+1+\
                      timedeltaprocess*24
        endrecord = startrecord + 24*numdays
    elif modelname == 'GOM3' and time_interval:
        timeperiod = starttime - (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
        startrecord=int(timeperiod.total_seconds()/60/60)
        endrecord=startrecord + 24*(time_interval.days)
    elif modelname == 'massbay' and time_interval:
        timeperiod = starttime - (datetime.now().replace(hour=0,minute=0)-timedelta(days=3))
        startrecord = int(timeperiod.total_seconds()/60/60)
        endrecord = startrecord + 24*(time_interval.days)
    else:
        raise Exception('You need to input time_interval if model name is massbay or GOM3')
    return startrecord, endrecord
#latsize=[39,45]
#lonsize=[-72.,-66]

'''
###############################################################################
def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)
    if event.button==3:
        latsize=[event.ydata-0.6,event.ydata+0.6]
        lonsize=[event.xdata-0.6,event.xdata+0.6]
        plt.figure(figsize=(7,6))
        m = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,\
            llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')#,fix_aspect=False)
        m.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,1),labels=[1,0,0,0])
        m.drawmeridians(np.arange(int(min(lonsize)),int(max(lonsize))+1,1),labels=[0,0,0,1])
        m.drawcoastlines()
        m.fillcontinents(color='grey')
        m.drawmapboundary()
        m.plot(lon,lat,'r.',lonc,latc,'b+')
        plt.show()
        spoint = pylab.ginput(1)
'''

def url_with_time_position(modelname, data):
    '''
    Get the data you want from certain model.

    modelname is the name of model, string.
    data stores the data wanted to get from web could be an array or a tuple.

    example of 'data'(Get u):
        if the data has several dimensions then:
            data = 'u[6][5][1:1:6]', 'v[6][5][1:1:45]', 'time[5][8]'
    '''
    string = ','.join(data)
    if modelname=='30yr':
        url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+string
    elif modelname == 'GOM3':
        url = "http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?"+string
    elif modelname == 'massbay':
        url = "http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?"+string
    else:
        raise Exception('Please use right model')
    return url

def get_coors(modelname, lo, la, lonc, latc, lon, lat, siglay, h, depth,startrecord, endrecord):
    if lo>90:
        [la,lo]=dm2dd(la,lo)
    print 'la, lo',la, lo
    latd,lond=[la],[lo]
    # kf,distanceF=nearlonlat(lonc,latc,lo,la) # nearest triangle center F - face
    # kv,distanceV=nearlonlat(lon,lat,lo,la)
    kf,distanceF = nearest_point_index(lo,la,lonc,latc,num=1)
    kv,distanceV = nearest_point_index(lo,la,lon,lat,num=1)
    kf = kf[0][0]
    kv = kv[0][0]
    print 'kf:', kf
    if h[kv] < 0:
        print 'Sorry, your position is on land, please try another point'
        sys.exit()
    depthtotal=siglay[:,kv]*h[kv]
    layer=np.argmin(abs(depthtotal-depth))
    for i in range(startrecord,endrecord):# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
############read the particular time model from website#########
        # print 'la, lo, i', la, lo, i
        timeurl='['+str(i)+':1:'+str(i)+']'
        uvposition=str([layer])+str([kf])
        data_want = ('u'+timeurl+uvposition, 'v'+timeurl+uvposition)
#        if urlname=="30yr":
#            url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
#                'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
#        elif urlname == "GOM3":
#            url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?"+\
#                'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
#        else:
#            url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?"+\
#                'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
        url = url_with_time_position(modelname, data_want)
        dataset = open_url(url)
        u=np.array(dataset['u'])
        v=np.array(dataset['v'])
        print 'u, v, i', u[0,0,0], v[0,0,0],i
################get the point according the position###################
        par_u=u[0,0,0]
        par_v=v[0,0,0]
        xdelta=par_u*60*60 #get_coors
        ydelta=par_v*60*60
        latdelta=ydelta/111111
        londelta=(xdelta/(111111*np.cos(la*np.pi/180)))
        la=la+latdelta
        lo=lo+londelta
        latd.append(la)
        lond.append(lo)
#        kf,distanceF=nearlonlat(lonc,latc,lo,la) # nearest triangle center F - face
#        kv,distanceV=nearlonlat(lon,lat,lo,la)# nearest triangle vertex V - vertex
        kf,distanceF = nearest_point_index(lo,la,lonc,latc,num=1)
        kv,distanceV = nearest_point_index(lo,la,lon,lat,num=1)
        kf, kv = kf[0][0], kv[0][0]
        depthtotal=siglay[:,kv]*h[kv]
#        layer=np.argmin(abs(depthtotal-depth))
        if distanceV>=0.3:
            if i==startrecord:
                print 'Sorry, your start position is NOT in the model domain'
                break
    return latd ,lond

def axes_interval(x):
    n = 0
    if 1<abs(x)<=10:
        n=1
    elif 10<abs(x)<180:
        n=10
    elif 0.1<abs(x)<=1:
        n=0.1
    elif 0.01<abs(x)<=0.1:
        n=0.01
    return n

###########save forecast in f[ID].dat file################
def write_data(file_opened, pointnum, TIME, latd, lond):
    time_trackpoints = [TIME]
    for i in range(pointnum):
        time_trackpoints.append(time_trackpoints[-1] + timedelta(hours=1))
        string = ('%s %s ' + str(latd[i]) + ' ' + str(lond[i]) + '\n')
        something = (str(time_trackpoints[0]), str(time_trackpoints[-1]))
        file_opened.seek(0, 2)           #This line have to be added in Windows()
#        file_open.write(('%s %s ' + str(latd[i]) + ' ' + str(lond[i]) + '\n') % (str(time_trackpoints[0]), str(time_trackpoints    1])))
        file_opened.write(string % something)

def save_data(pointnum, TIME, latd, lond):
    f = open('f%s.dat' % ID,'a+')
    if len(f.read()) == 0:
        f.write('startdate' + '  ' + 'date/time' + ' ' + 'lat' + ' ' + 'lon\n')
        write_data(f, pointnum, TIME, latd, lond)
    else:
        write_data(f, pointnum, TIME, latd, lond)
    f.write('\n')
    f.close()
def on_left_click_zoomin(event):
    if event.button == 1 and event.xdata and event.ydata:
        x, y = event.xdata, event.ydata
        print 'You clicked: ', event.button, x, y
        lat = [y - 0.6,y + 0.6]
        lon = [x - 0.6,x + 0.6]
        fig, ax = draw_figure(lat, lon)
#        fig = plt.figure()
#        ax = plt.add_subplot(111)
        plt.title('zoomin figure')
        fig.canvas.mpl_connect('button_press_event', on_left_click_show)
        fig.show()
#    else:
#        print 'Please press left mouse button in the map area1'
#def on_left_click_show(event):
#    if event.button == 1 and event.xdata and event.ydata:
#        x, y = event.xdata, event.ydata
#        print "You clicked: ", x, y
#        latd, lond = get_coors(modelname, x, y, lonc, latc, lon, lat, siglay, h, depth, startrecord, endrecord)
#        pointnum = len(latd)
#        save_data(pointnum, TIME, lat, lond)
#        extra_lat=(max(latd)-min(latd))/10.
#        extra_lon=(max(lond)-min(lond))/10.
#        latsize=[min(latd)-extra_lat,max(latd)+extra_lat]
#        lonsize=[min(lond)-extra_lon,max(lond)+extra_lon]
#        fig, ax = draw_figure(latsize, lonsize, interval_lat=axes_interval(max(latd)-min(latd)),interval_lon=axes_interval(max(lond)-min(lond)))
#        xy = (lond[0] ,latd[0])
#        xytext = (lond[0]+.5*dist_cmp(lond[0], lonsize[0], lonsize[1]), latd[0]+.5*dist_cmp(latd[0], latsize[0], latsize[1]))
#        plt.annotate('Startpoint', xy, xytext = (lond[0]+.5*dist_cmp(lond[0], lonsize[0], lonsize[1]),
#                     latd[0]+.5*dist_cmp(latd[0], latsize[0], latsize[1])),
#                     arrowprops = dict(arrowstyle = 'simple'))
#        plt.plot(lond,latd,'ro-',lond[-1],latd[-1],'mo',lond[0],latd[0],'mo')
#        plt.title(modelname+' model track Depth:'+str(depth)+' Time:'+str(TIME))
#        plt.savefig(modelname+'driftrack.png', dpi = 200)
#        plt.show()
#    else:
#        print "Please press left mouse button in the map area2"
def on_left_button_down(event):
    if event.button == 1 and event.xdata and event.ydata:
        x, y = event.xdata, event.ydata
        print "You clicked: ", x, y
        if event.inaxes.title.get_text() == 'zoomin figure':
            latd, lond = get_coors(modelname, x, y, lonc, latc, lon, lat, siglay, h, depth, startrecord, endrecord)
            pointnum = len(latd)
            save_data(pointnum, TIME, lat, lond)
            extra_lat=(max(latd)-min(latd))/10.
            extra_lon=(max(lond)-min(lond))/10.
            latsize=[min(latd)-extra_lat,max(latd)+extra_lat]
            lonsize=[min(lond)-extra_lon,max(lond)+extra_lon]
            fig, ax = draw_figure(latsize, lonsize, interval_lat=axes_interval(max(latd)-min(latd)),interval_lon=axes_interval(max(lond)-min(lond)))
            xy = (lond[0] ,latd[0])
            xytext = (lond[0]+.5*dist_cmp(lond[0], lonsize[0], lonsize[1]), latd[0]+.5*dist_cmp(latd[0], latsize[0], latsize[1]))
            plt.annotate('Startpoint', xy, xytext = (lond[0]+.5*dist_cmp(lond[0], lonsize[0], lonsize[1]),
                         latd[0]+.5*dist_cmp(latd[0], latsize[0], latsize[1])),
                         arrowprops = dict(arrowstyle = 'simple'))
            plt.plot(lond,latd,'ro-',lond[-1],latd[-1],'mo',lond[0],latd[0],'mo')
            plt.title(modelname+' model track Depth:'+str(depth)+' Time:'+str(TIME))
            plt.savefig(modelname+'driftrack.png', dpi = 200)
            plt.show()
        elif event.inaxes.title.get_text() == name_1st_figure:
            lat_range = [y - 0.6,y + 0.6]
            lon_range = [x - 0.6,x + 0.6]
            fig, ax = draw_figure(lat_range, lon_range)
#            fig = plt.figure()
#            ax = plt.add_subplot(111)
            plt.title('zoomin figure')
            fig.canvas.mpl_connect('button_press_event', on_left_button_down)
            print 'get into on_left_button_down'
            fig.show()
def draw_figure(latsize, lonsize, interval_lat = 1, interval_lon = 1):
    '''
    draw the Basemap
    '''
#    p = plt.figure(figsize = (7, 6))
    fig = plt.figure()
    dmap = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')
    dmap.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,interval_lat),labels=[1,0,0,0])
    dmap.drawmeridians(np.arange(int(min(lonsize))-1,int(max(lonsize))+1,interval_lon),labels=[0,0,0,1])
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
    return fig, dmap

def dist_cmp(v, v1, v2):
    """
    compare the distance from v to v1 and v2, return the nearer one.
    """
    d1 = v1 - v
    d2 = v2 - v
    if abs(d1) > abs(d2):
        return d2
    else:
        return d1
#def draw_map_click():
#    fig1, m1 = draw_figure(lat,lon)
#    cid1 = fig1.canvas.mpl_connect('button_press_event', on_press)
#    plt.title(urlname+' model track Depth:'+str(depth)+' Time:'+str(TIME))
#    plt.savefig(urlname+'driftrack.png', dpi = 200)
#    plt.show()

######### HARDCODES ########
print 'This routine reads a control file called ctrl_trackzoomin.csv'
modelname=open("ctrl_trackzoomin.csv", "r").readlines()[0][37:-1]
depth=int(open("ctrl_trackzoomin.csv", "r").readlines()[1][22:-1])
TIME=open("ctrl_trackzoomin.csv", "r").readlines()[2][31:-1]
numdays=int(open("ctrl_trackzoomin.csv", "r").readlines()[3][24:-1])
TIME=datetime.strptime(TIME, "%Y-%m-%d %H:%M:%S")
methods_get_startpoint = open("ctrl_trackzoomin.csv", "r").readlines()[4][35:-1]
ID = int(input_with_default('ID', 130400681))
datetime_today = datetime.now().replace(hour=0, minute=0, second=0,microsecond=0)

if modelname=="massbay" or "GOM3":
    time_interval = timedelta(days=numdays)
    if TIME+time_interval>datetime_today+timedelta(days=3) or\
       TIME+time_interval<datetime_today-timedelta(days=3):
        sys.exit("please check your numday.access period is in [now-3days,now+3days]")
    startrecord, endrecord = get_indices(modelname, TIME, time_interval)
    data = ('lon', 'lat', 'latc', 'lonc', 'siglay', 'h',
            'Times['+str(startrecord)+':1:'+str(startrecord)+']')
    url = url_with_time_position(modelname, data)
elif modelname == "30yr":
    startrecord, endrecord = get_indices(modelname, TIME)
    data = ('lon', 'lat', 'latc', 'lonc', 'siglay', 'h',
            'Times['+str(startrecord)+':1:'+str(startrecord)+']')
    url = url_with_time_position(modelname, data)

dataset = netCDF4.Dataset(url)
latc = np.array(dataset.variables['latc'][:])
lonc = np.array(dataset.variables['lonc'][:])
lat = np.array(dataset.variables['lat'][:])
lon = np.array(dataset.variables['lon'][:])
siglay=np.array(dataset.variables['siglay'][:])
h=np.array(dataset.variables['h'][:])
#dataset = open_url(url)
#latc = np.array(dataset['latc'])
#lonc = np.array(dataset['lonc'])
#lat = np.array(dataset['lat'])
#lon = np.array(dataset['lon'])
#siglay=np.array(dataset['siglay'])
#h=np.array(dataset['h'])

if methods_get_startpoint == "input":
    la = float(input_with_default('lat', 41.433))
    lo = float(input_with_default('lon', -69.258))
    latd, lond = get_coors(modelname, lo, la, lonc, latc, lon, lat,
                           siglay, h, depth, startrecord, endrecord)
    pointnum = len(latd)
    # save_data(pointnum, TIME, latd, lond)
    extra_lat=(max(latd)-min(latd))/10.
    extra_lon=(max(lond)-min(lond))/10.
    latsize=[min(latd)-extra_lat-1,max(latd)+extra_lat+1]
    lonsize=[min(lond)-extra_lon-1,max(lond)+extra_lon+1]
    fig2, ax2 = draw_figure(latsize, lonsize)
    plt.annotate('Startpoint',
                 xytext=(lond[0]+.5*dist_cmp(lond[0], lonsize[0], lonsize[1]),
                         latd[0]+.5*dist_cmp(latd[0], latsize[0], latsize[1])),
                 xy=(lond[0] ,latd[0]),
                 arrowprops = dict(arrowstyle = 'simple'))
    ax2.plot(lond,latd,'ro-',lond[-1],latd[-1],'mo',lond[0],latd[0],'mo')
    plt.title(modelname+' model track Depth:'+str(depth)+' Time:'+str(TIME))
    plt.savefig(modelname+'driftrack.png', dpi = 200)
    plt.show()
elif methods_get_startpoint == "click":
#    draw_map_click()
    fig, ax = draw_figure(lat,lon)
    cid1 = fig.canvas.mpl_connect('button_press_event', on_left_button_down)
    name_1st_figure = modelname+' model track Depth:'+str(depth)+' Time:'+str(TIME)
    plt.title(name_1st_figure)
    plt.savefig(modelname+'driftrack.png', dpi = 200)
    plt.show()
else:
    print "Please check your control file if 'ways_coors_get' is right"
