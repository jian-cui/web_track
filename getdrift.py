# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:35:02 2013

@author: jmanning
"""
import sys
import numpy as np
from pydap.client import open_url
from utilities import nearxy
import datetime as dt
from matplotlib.dates import num2date, date2num
from matplotlib.ticker import FormatStrFormatter
import matplotlib.mlab as ml
import scipy
import time
import matplotlib.pyplot as plt

def getdrift_header():
    # simple function that returns all ids in the drift_header table
    url = 'http://gisweb.wh.whoi.edu:8080/dods/whoi/drift_header'
    dataset = get_dataset(url)
    id=map(int,dataset.drift_header.ID)
    return id

def getdrift_ids():
    # simple function that returns all distinct ids in the drift_data table
    # this takes a few minutes and is limited to 300000 until the server is
    # restarted to pick up a 100000 "JDBCMaxResponseLength in web_sv.xml
    url = 'http://gisweb.wh.whoi.edu:8080/dods/whoi/drift_data'
    dataset = get_dataset(url)
    print 'Note: It takes a minute or so to determine distinct ids'
    ids=list(set(list(dataset.drift_data.ID)))
    return ids

def getdrift(id):
    """
    uses pydap to get remotely stored drifter data given an id number
    """
    print 'using pydap'
    url = 'http://gisweb.wh.whoi.edu:8080/dods/whoi/drift_data'

    dataset = get_dataset(url)

    try:
        lat = list(dataset.drift_data[dataset.drift_data.ID == id].LAT_DD)
    except:
        print 'Sorry, ' + str(id) + ' is not available'
        sys.exit(0)

    lon = list(dataset.drift_data[dataset.drift_data.ID == id].LON_DD)
    year_month_day = list(dataset.drift_data[dataset.drift_data.ID == id].TIME_GMT)
    yearday = list(dataset.drift_data[dataset.drift_data.ID == id].YRDAY0_GMT)
    dep = list(dataset.drift_data[dataset.drift_data.ID == id].DEPTH_I)
    datet = []
    # use time0('%Y-%m-%d) and yearday to calculate the datetime
    for i in range(len(yearday)):
        #datet.append(num2date(yearday[i]).replace(year=time.strptime(time0[i], '%Y-%m-%d').tm_year).replace(day=time.strptime(time0[i], '%Y-%m-%d').tm_mday))
        datet.append(num2date(yearday[i]+1.0).replace(year=dt.datetime.strptime(year_month_day[i], '%Y-%m-%d').year).replace(month=dt.datetime.strptime(year_month_day[i],'%Y-%m-%d').month).replace(day=dt.datetime.strptime(year_month_day[i],'%Y-%m-%d').day).replace(tzinfo=None))

    print 'Sorting drifter data by time'
    index = range(len(datet))
    index.sort(lambda x, y:cmp(datet[x], datet[y]))
    #reorder the list of date_time,u,v
    datet = [datet[i] for i in index]
    lat = [lat[i] for i in index]
    lon = [lon[i] for i in index]
    dep=[dep[i] for i in index]
    return lat, lon, datet, dep

def getdrift_raw(filename,id3,interval,datetime_wanted,num = None,step_size = None):
  # range_time is a number,unit by one day.  datetime_wanted format is num
  d=ml.load(filename)
  lat1=d[:,8]
  lon1=d[:,7]
  idd=d[:,0]
  year=[]
  for n in range(len(idd)):
      year.append(str(idd[n])[0:2])
  h=d[:,4]
  day=d[:,3]
  month=d[:,2]
  time1=[]
  for i in range(len(idd)):
      time1.append(date2num(dt.datetime.strptime(str(int(h[i]))+' '+str(int(day[i]))+' '+str(int(month[i]))+' '+str(int(year[i])), "%H %d %m %y")))

  idg1=list(ml.find(idd==id3))
  if not num:
      idg2=list(ml.find(np.array(time1)<=datetime_wanted+interval/24))
      idg3=list(ml.find(np.array(time1)>=datetime_wanted-0.1))
  else:
      idg2=list(ml.find(np.array(time1)<=datetime_wanted+step_size/24.0*(num-1)+0.25))
      idg3=list(ml.find(np.array(time1)>=datetime_wanted-interval/24.0))
  "'0.25' means the usual Interval, It can be changed base on different drift data "
  idg3=list(ml.find(np.array(time1)>=datetime_wanted-0.1))
  idg23=list(set(idg2).intersection(set(idg3)))
  # find which data we need
  idg=list(set(idg23).intersection(set(idg1)))
  print 'the length of drifter data is  '+str(len(idg)),str(len(set(idg)))+'   . if same, no duplicate'
  lat,lon,time=[],[],[]

  for x in range(len(idg)):
      lat.append(round(lat1[idg[x]],4))
      lon.append(round(lon1[idg[x]],4))
      time.append(round(time1[idg[x]],4))
  # time is num
  return lat,lon,time
  if not num:
      return lat,lon,time
  else:
      maxlon=max(lon)
      minlon=min(lon)
      maxlat=max(lat)
      minlat=min(lat)
      return maxlon,minlon,maxlat,minlat
