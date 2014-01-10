# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:15:52 2013
This toutine reads a control file called ctrl_trackzoomin.csv
@author: jmanning
"""
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
#import pylab
from datetime import datetime
from pydap.client import open_url
from datetime import timedelta
from conversions import dm2dd
import sys

######### HARDCODES ########
print 'This routine reads a control file called ctrl_trackzoomin.csv'
urlname=open("ctrl_trackzoomin.csv", "r").readlines()[0][37:-1]
depth=int(open("ctrl_trackzoomin.csv", "r").readlines()[1][22:-1])
TIME=open("ctrl_trackzoomin.csv", "r").readlines()[2][31:-1]
#TIME = datetime.now()
numdays=int(open("ctrl_trackzoomin.csv", "r").readlines()[3][24:-1])
TIME=datetime.strptime(TIME, "%Y-%m-%d %H:%M:%S")
new_numdays=timedelta(days=numdays)
datetime_today = datetime.now().replace(hour=0, minute=0, second=0,microsecond=0)
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
ID = int(input_with_default('ID', 130400681))
la = float(input_with_default('lat', 4015.497))
lo = float(input_with_default('lon', 6901.6878))
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

#def dist(lon,lat,lonp,latp):
#    r = 6378.1
#    dist = 2 * (r *2) * (1 - np.cos(lat)*np.cos(latp)*np.cos(lon-lonp) + \
#                        np.sin(lat)*np.sin(latp))
#    return dist
#
#def nearestdist(lon,lat,lonp,latp):
#    dist = dist(lon,lat,lonp,latp)
#    i = np.argmin(dist)
#    min_dist = np.sqrt(dist[i])
#    return i,min_dist

def days3_judge(TIME, days):
    '''
    when model is massbay or GOM3, judge if the time user input is in and before 3 days
    -TIME: startime
    -days: days = timedelta(days=numdays)
    '''
    # date_today = datetime.now().replace(hour=0, minute=0, second=0,microsecond=0)
    if TIME+days>datetime_today+timedelta(days=3) or TIME-days<datetime_today+timedelta(days=3):
        sys.exit("please check your numday.access period is in [now-3days,now+3days]")

#latsize=[39,45]
#lonsize=[-72.,-66]

'''
m = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,\
            llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')#,fix_aspect=False)
m.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,1),labels=[1,0,0,0])
m.drawmeridians(np.arange(int(min(lonsize)),int(max(lonsize))+1,1),labels=[0,0,0,1])
m.drawcoastlines()
m.fillcontinents(color='grey')
m.drawmapboundary()
'''
def record_range(TIME, new_numdays, date_now):
    timeperiod=(TIME+new_numdays)-(date_now-timedelta(days=3))
    startrecord=(timeperiod.seconds)/60/60
    endrecord=startrecord+24*(new_numdays.days)
    return startrecord, endrecord
if urlname == '30yr':
#    stime=datetime.strptime(TIME, "%Y-%m-%d %H:%M:%S")
    timesnum=TIME.year-1981
    standardtime=datetime.strptime(str(TIME.year)+'-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
    timedeltaprocess=(TIME-standardtime).days
    startrecord=26340+35112*(timesnum/4)+8772*(timesnum%4)+1+timedeltaprocess*24 # note: 26340 is the model time index for Jan 1, 1981
    endrecord=startrecord+24*numdays
#    url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+ \
#        'lon,lat,latc,lonc,siglay,h,Times['+str(startrecord)+':1:'+str(startrecord)+']'
    url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+ \
        'lon,lat,latc,lonc,siglay,h'
elif urlname == 'GOM3':
    days3_judge(TIME, new_numdays)
    startrecord, endrecord = record_range(TIME, new_numdays, datetime_today)
    url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?"+\
        'lon,lat,latc,lonc,siglay,h,Times['+str(startrecord)+':1:'+str(startrecord)+']'
elif urlname == 'massbay':
    days3_judge(TIME, new_numdays)
    startrecord, endrecord = record_range(TIME, new_numdays, datetime_today)
    url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?"+\
        'lon,lat,latc,lonc,siglay,h,Times['+str(startrecord)+':1:'+str(startrecord)+']'

    
dataset = open_url(url)
latc = np.array(dataset['latc'])
lonc = np.array(dataset['lonc'])
lat = np.array(dataset['lat'])
lon = np.array(dataset['lon'])
siglay=np.array(dataset['siglay'])
h=np.array(dataset['h'])

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

if lo>90:
    [la,lo]=dm2dd(la,lo)
latd,lond=[],[]

kf,distanceF=nearlonlat(lonc,latc,lo,la) # nearest triangle center F - face
kv,distanceV=nearlonlat(lon,lat,lo,la)

if h[kv] < 0:
    print 'Sorry, your position is on land, please try another point'
    sys.exit()

depthtotal=siglay[:,kv]*h[kv]
layer=np.argmin(abs(depthtotal-depth))

for i in range(startrecord,endrecord):
############read the particular time model from website#########
               timeurl='['+str(i)+':1:'+str(i)+']'
               uvposition=str([layer])+str([kf])
               if urlname == "30yr":
                   url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+\
                       'Times'+timeurl + ',u' + timeurl + uvposition + ',' + 'v'+timeurl+uvposition
               elif urlname == "GOM3":
                   url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc?"+\
                       'Times'+timeurl + ',u' + timeurl + uvposition + ',' + 'v'+timeurl+uvposition
               elif urlname == "massbay":
                   url="http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?"+\
                       'Times'+timeurl + ',u' + timeurl + uvposition + ',' + 'v'+timeurl+uvposition

               datasetuv = open_url(url)
               u=np.array(datasetuv['u'])
               v=np.array(datasetuv['v'])
################get the point according the position###################min(lonsize)
#               print kf,u[0,0,0],v[0,0,0],layer
               par_u=u[0,0,0]
               par_v=v[0,0,0]
               xdelta=par_u*60*60
               ydelta=par_v*60*60
               latdelta=ydelta/111111
               londelta=(xdelta/(111111*np.cos(la*np.pi/180)))
               la=la+latdelta
               lo=lo+londelta
               latd.append(la)
               lond.append(lo)
               kf,distanceF=nearlonlat(lonc,latc,lo,la) # nearest triangle center F - face
               kv,distanceV=nearlonlat(lon,lat,lo,la)# nearest triangle vertex V - vertex
               depthtotal=siglay[:,kv]*h[kv]
#               layer=np.argmin(abs(depthtotal-depth))
               if distanceV>=0.3:
                   if i==startrecord:
                      print 'Sorry, your start position is NOT in the model domain'
                   break

def axes_interval(x):
    n=0
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
pointnum = len(latd)
def save_data(pointnum, TIME, lat, lond):
    f = open('f%s.dat' % ID,'a+')
    if len(f.read()) == 0:
        f.write('startdate' + '  ' + 'date/time' + ' ' + 'lat' + ' ' + 'lon\n')
        write_data(f, pointnum, TIME, latd, lond)
    else:
        write_data(f, pointnum, TIME, latd, lond)
    f.write('\n')
    f.close()

############draw pic########################
#plt.figure()
extra_lat=[(max(latd)-min(latd))/10.]
extra_lon=[(max(lond)-min(lond))/10.]
latsize=[min(latd)-extra_lat,max(latd)+extra_lat]
lonsize=[min(lond)-extra_lon,max(lond)+extra_lon]

#def on_press(event):
#    if event.button == 1 and event.xdata != None and event.ydata != None:
#        x, y = event.xdata, event.ydata
#        print 'You clicked: ', event.button, x, y
#        lat = [y - 0.6, y + 0.6]
#        lon = [x + 0.6, x + 0.6]
#        fig2, m2 = draw_figure(lat, lon)
#        cid2 = fig2.canvas.mpl_connect('button_press_event', on_press2)
#        plt.annotate('Startpoint',xytext=(lond[0]+axes_interval(max(lond)-min(lond)),\
#                     latd[0]+axes_interval(max(latd)-min(latd))),xy=(lond[0] ,latd[0]),\
#                     arrowprops = dict(arrowstyle = 'simple'))
#        plt.plot(lond,latd,'ro-',lond[-1],latd[-1],'mo',lond[0],latd[0],'mo')
#    else:
#        print 'Please press left mouse button in the map area'
#def on_press2(event):
#    if event.button == 1 and event.xdata != None and event.ydata != None:
#        x, y = event.xdata, event.ydata
#        print "You clicked: ", x, y
#
#    else:
#        print "Please press left mouse button in the map area"
def draw_figure(latsize, lonsize):
#    p = plt.figure(figsize = (7, 6))
    p = plt.figure()
    ax = p.add_subplot(111)
    dmap = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,
            llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')
    dmap.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,1),labels=[1,0,0,0])
    dmap.drawmeridians(np.arange(int(min(lonsize)),int(max(lonsize))+1,1),labels=[0,0,0,1])
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
    return p,ax
#fig = plt.figure()
#m = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,\
#  llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')#,fix_aspect=False)
##m.drawparallels(np.arange(round(min(latsize)-1, 0),round(max(latsize)+1, 0),1),labels=[1,0,0,0])
##m.drawmeridians(np.arange(round(min(lonsize)-1, 2),round(max(lonsize)+1, 2),\
##                axes_interval(max(lond)-min(lond))),labels=[0,0,0,1])
#m.drawparallels(np.arange(round(min(latsize), 0),round(max(latsize)+1, 0),1),labels=[1,0,0,0])
#m.drawmeridians(np.arange(round(min(lonsize), 0),round(max(lonsize)+1, 0),\
#                axes_interval(max(lond)-min(lond))),labels=[0,0,0,1])
#m.drawcoastlines()
#m.fillcontinents(color='blue')
#m.drawmapboundary()
fig,ax = draw_figure(latsize, lonsize)
#cid = fig.canvas.mpl_connect('button_press_event', on_press)
'''
m.plot(lon,lat,'r.',lonc,latc,'b+')
fig=plt.figure(figsize=(7,6))
plt.plot(lon,lat,'r.',lonc,latc,'b+')
'''
#plt.annotate('Startpoint',xytext = (lond[0]+0.01, latd[0]), xy = (lond[0] ,latd[0]), arrowprops = dict(arrowstyle = 'simple'))
#ax.annotate('Startpoint',xytext=(lond[0]+axes_interval(max(lond)-min(lond)),\
#             latd[0]+axes_interval(max(latd)-min(latd))),xy=(lond[0] ,latd[0]),\
#             arrowprops = dict(arrowstyle = 'simple'))
def dist_comp(v, v1, v2):
    """
    compare the distance from v to v1 or v2, return the nearer one.
    """
    d1 = v1 - v
    d2 = v2 - v
    if abs(d1) > abs(d2):
        return d2
    else:
        return d1
ax.annotate('Startpoint',xytext=(lond[0]+.5*dist_comp(lond[0], lonsize[0], lonsize[1]),\
             latd[0]+.5*dist_comp(latd[0], latsize[0], latsize[1])),xy=(lond[0] ,latd[0]),\
             arrowprops = dict(arrowstyle = 'simple'))
ax.plot(lond,latd,'ro-',lond[-1],latd[-1],'mo',lond[0],latd[0],'mo')
plt.title(urlname+' model track Depth:'+str(depth)+' Time:'+str(TIME))
plt.savefig(urlname+'driftrack.png', dpi = 200)
plt.show()
'''
return True
cid= fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
'''
