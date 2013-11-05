# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:09:30 2013

@author: jmanning
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pylab
from datetime import datetime
from pydap.client import open_url
from datetime import timedelta
import sys
import pandas as pd
import os
def RungeKutta4_lonlat(lon,lat,Grid,u,v,tau):       
    lon1=lon*1.;          lat1=lat*1.;        urc1,v1=VelInterp_lonlat(lon1,lat1,Grid,u,v);  
    lon2=lon+0.5*tau*urc1;lat2=lat+0.5*tau*v1;urc2,v2=VelInterp_lonlat(lon2,lat2,Grid,u,v);
    lon3=lon+0.5*tau*urc2;lat3=lat+0.5*tau*v2;urc3,v3=VelInterp_lonlat(lon3,lat3,Grid,u,v);
    lon4=lon+    tau*urc3;lat4=lat+    tau*v3;urc4,v4=VelInterp_lonlat(lon4,lat4,Grid,u,v);
    lon=lon+tau/6.*(urc1+2.*urc2+2.*urc3+urc4);
    lat=lat+tau/6.*(v1+2.*v2+2.*v3+v4); 
    uinterplation=  (urc1+2.*urc2+2.*urc3+urc4)/6
    vinterplation= (v1+2.*v2+2.*v3+v4)/6
   # print urc1,v1,urc2,v2,urc3,v3,urc4,v4
    return lon,lat,uinterplation,vinterplation   
def nearxy(x,y,xp,yp):
    dx=x-xp
    dy=y-yp
    dist2=dx*dx+dy*dy
   # dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
    return i
def nearlonlat(lon,lat,lonp,latp):
    cp=np.cos(latp*np.pi/180.)
# approximation for small distance
    dx=(lon-lonp)*cp
    dy=lat-latp
    dist2=dx*dx+dy*dy
# dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
    min_dist=np.sqrt(dist2[i])
    return i,min_dist     
def polygonal_barycentric_coordinates(xp,yp,xv,yv):
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N
    jb=(j-1)%N
    Ajab=np.cross(np.array([xv[ja]-xv[j],yv[ja]-yv[j]]).T,np.array([xv[jb]-xv[j],yv[jb]-yv[j]]).T)
    Aj=np.cross(np.array([xv[j]-xp,yv[j]-yp]).T,np.array([xv[ja]-xp,yv[ja]-yp]).T)
    Aj=abs(Aj)
    Ajab=abs(Ajab)
    Aj=Aj/max(abs(Aj))
    Ajab=Ajab/max(abs(Ajab))    
    w=xv*0.
    j2=np.arange(N-2)
    for j in range(N):
        w[j]=Ajab[j]*Aj[(j2+j+1)%N].prod()
      #  print Ajab[j],Aj[(j2+j+1)%N]
    w=w/w.sum()
    return w
def VelInterp_lonlat(lonp,latp,Grid,u,v):    
# find the nearest vertex    
    kv,distance=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
 #   print kv,lonp,latp
# list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
  #  print kfv
# coordinates of the (dual mesh) polygon vertices: the centers of triangle faces
    lonv=Grid['lonc'][kfv];latv=Grid['latc'][kfv] 
    w=polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
# baricentric coordinates are invariant wrt coordinate transformation (xy - lonlat), check!    
#    print w
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
    
def rddate(TIME,numdays):    
    stime=datetime.strptime(TIME, "%Y-%m-%d %H:%M:%S")
    timesnum=stime.year-1981
    standardtime=datetime.strptime(str(stime.year)+'-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
    timedeltaprocess=(stime-standardtime).days
    startrecord=26340+35112*(timesnum/4)+8772*(timesnum%4)+1+timedeltaprocess*24
    endrecord=startrecord+24*numdays
    return startrecord,endrecord,stime
def get_uv_web(time,layer):
    timeurl='['+str(time)+':1:'+str(time)+']'
    uvposition=str([layer])+'[0:1:90414]'
    url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
    dataset = open_url(url)
    utotal=np.array(dataset['u'])
    vtotal=np.array(dataset['v'])
    times=np.array(dataset['Times'])
    u=utotal[0,0,:]
    v=vtotal[0,0,:]
    print times,layer
    return u,v
def draw_figure(latsize, lonsize):
#    p = plt.figure(figsize = (7, 6))
    p = plt.figure()
    dmap = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,\
            llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')
    dmap.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,1),labels=[1,0,0,0])
    dmap.drawmeridians(np.arange(int(min(lonsize)),int(max(lonsize))+1,1),labels=[0,0,0,1])
    dmap.drawcoastlines()
    dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
    return p,dmap
def on_right_click(event):
    if event.button == 3:
        print 'you pressed: ', event.button, event.xdata, event.ydata
        x = event.xdata
        y = event.ydata
        pic_zoom_in(x, y)
    else:
        print 'please press the map with right button'
    return True
def on_left_click(event):
    if event.button == 1:
        print 'you clicked: ', event.button, event.xdata,event.ydata
        x = event.xdata
        y = event.ydata
        pic_trend(x, y)
    else:
        print "Please press the map with right button"
    return True
def pic_zoom_in(x,y):
    latsize=[y - 0.6,y + 0.6]
    lonsize=[x - 0.6,x + 0.6]
    p, m = draw_figure(latsize, lonsize)
    m.plot(lon,lat,'r.',lonc,latc,'b+')
#    m.plot(lon,lat,'r.',lonc,latc,'b+')
    surface_cid = p.canvas.mpl_connect('button_press_event', on_left_click)
    plt.show()
def pic_trend(lond, latd):
    dt=60*60.
    tau=dt/111111.
    lont=[]
    latt=[]
    ufinal=[]
    vfinal=[]
    for i in range(startrecord,endrecord):
        timeurl = '['+str(i)+':1:'+str(i)+']'
        uvposition = str([0])+'[0:1:90414]' # this is the number of grid points in thie 30yr model
        url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
        dataset = open_url(url)
        utotal=np.array(dataset['u'])
        vtotal=np.array(dataset['v'])
        times=np.array(dataset['Times'])
        u=utotal[0,0,:]
        v=vtotal[0,0,:]
        lont.append(lond)
        latt.append(latd)
        lond,latd,uinterplation,vinterplation=RungeKutta4_lonlat(lond,latd,Grid,u,v,tau)
        ufinal.append(uinterplation)
        vfinal.append(vinterplation)
        kv,distance=nearlonlat(lon,lat,lond,latd)
        if distance>=0.3:
            break
    fig=plt.figure()     
    Q=plt.quiver(lont,latt,ufinal,vfinal,scale=5.)  
    plt.show() 
TIME='2003-01-08 00:00:00' 
numdays=30 
#lond=-67
#latd=42
depth=0
latsize=[39,45]
lonsize=[-72.,-66]
model='30yr'
'''
fig=plt.figure(figsize=(7,6))
m = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,\
            llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')#,fix_aspect=False)
m.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,1),labels=[1,0,0,0])
m.drawmeridians(np.arange(int(min(lonsize)),int(max(lonsize))+1,1),labels=[0,0,0,1])
m.drawcoastlines()
m.fillcontinents(color='grey')
m.drawmapboundary()
'''
startrecord,endrecord,stime=rddate(TIME,numdays)

url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+'lon,lat,latc,lonc,siglay,h,x,y,xc,yc,nv,nbe,nbsn,ntsn,nbve,ntve,Times['+str(startrecord)+':1:'+str(endrecord)+']'
dataset = open_url(url)
latc = np.array(dataset['latc'])
lonc = np.array(dataset['lonc'])
lat = np.array(dataset['lat'])
lon = np.array(dataset['lon'])
x=np.array(dataset['x'])
y=np.array(dataset['y'])
xc=np.array(dataset['xc'])
yc=np.array(dataset['yc'])
nv=np.array(dataset['nv'])
nbe=np.array(dataset['nbe'])
nbsn=np.array(dataset['nbsn'])
ntsn=np.array(dataset['ntsn'])
nbve=np.array(dataset['nbve'])
ntve=np.array(dataset['ntve'])
siglay=np.array(dataset['siglay'])
h=np.array(dataset['h'])
coslat=np.cos(lat*np.pi/180.)
coslatc=np.cos(latc*np.pi/180.)
#################ready to process############################
Grid={'x':x,'y':y,'xc':xc,'yc':yc,'lon':lon,'lat':lat,'lonc':lonc,'latc':latc,'coslat':coslat,'coslatc':coslatc,'kvf':nv,'kff':nbe,'kvv':nbsn,'nvv':ntsn,'kfv':nbve,'nfv':ntve}
#kf=nearlonlat(lonc,latc,lond,latd) # nearest triangle center F - face
###########################draw the basic map############################################
fig, m = draw_figure(lat, lon)
m.plot(lon,lat,'r.',lonc,latc,'b+')
cid = fig.canvas.mpl_connect('button_press_event', on_right_click)
plt.show()


#latsize=[coor[0].y - 0.6, coor[0].y + 0.6]
#lonsize=[coor[0].x - 0.6, coor[0].x + 0.6]
#p, n = draw_figure(latsize, lonsize)
#n.plot(lon,lat,'r.',lonc,latc,'b+')
#plt.show()

#spoint = pylab.ginput(1)
#latd=spoint[0][1]
#lond=spoint[0][0]
##depthtotal=siglay[:,kv]*h[kv]
##layer=np.argmin(abs(depthtotal-depth))
#dt=60*60.
#tau=dt/111111.
#lont=[]
#latt=[]
#ufinal=[]
#vfinal=[]
#for i in range(startrecord,endrecord):
##    u,v=get_uv_web(i,layer=0)
#    timeurl = '['+str(i)+':1:'+str(i)+']'
##    uvposition=str([layer])+str([kf])
#    if model == 'massbay':        
#        uvposition = str([0])+'[0:1:90414]' # this is the number of grid points in thie 30yr model
#        url='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
#    elif model == '30yr':     
#        uvposition = str([0])+'[0:1:90414]' # this is the number of grid points in thie 30yr model
#        url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?'+'Times'+timeurl+',u'+timeurl+uvposition+','+'v'+timeurl+uvposition
#    else:
#        sys.exit('Please input right model name.')
#    dataset = open_url(url)
#    utotal=np.array(dataset['u'])
#    vtotal=np.array(dataset['v'])
#    times=np.array(dataset['Times'])
#    u=utotal[0,0,:]
#    v=vtotal[0,0,:]
#################get the point according the position###################
#    lont.append(lond)
#    latt.append(latd)
#    lond,latd,uinterplation,vinterplation=RungeKutta4_lonlat(lond,latd,Grid,u,v,tau)
##    print lond,latd,times,uinterplation,vinterplation
#    ufinal.append(uinterplation)
#    vfinal.append(vinterplation)
#    kv,distance=nearlonlat(lon,lat,lond,latd)
##    print distance
#    if distance>=0.3:
#         break
#    
#    
##fig=plt.figure(figsize=(7,6))    
##plt.plot(lon,lat,'r.',lonc,latc,'b+')
##plt.plot(lont,latt,'ro-',lont[-1],latt[-1],'mo',lont[0],latt[0],'mo')
##
###kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
###plt.plot(Grid['lonc'][kfv],Grid['latc'][kfv],'gd')
##plt.title('30yr model map surface track Depth:-1  '+' Time:'+TIME) 
##plt.show()
#fig=plt.figure()     
#Q=plt.quiver(lont,latt,ufinal,vfinal,scale=5.)  
#plt.show() 
#'''
#rng = pd.date_range(stime, periods=len(ufinal), freq='H')
#dfu = pd.DataFrame(ufinal, index=rng,columns=['u'])
#dfv=pd.DataFrame(vfinal, index=rng,columns=['v'])
#dfu.plot()
#dfv.plot()
#plt.show()
#'''

































