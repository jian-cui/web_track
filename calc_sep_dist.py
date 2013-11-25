'''
created on Nov 21, 11:20
compare data forecasted with data recorded by drifter.
'''
import xlrd
import xlwt
from conversions import dist
from datetime import datetime

def input_with_default(data, default):
    '''
    return a str
    '''
    l = (data, str(default))
    data_input = raw_input('Please input %s(default %s): ' % l)
    if data_input == '':
        data_input = l[1]
    else:
        data_input = data_input
    return data_input
    
def pos_real(f, time, ID):
    time_real = []
    coor = []  
    for line in f.readlines():
        data_temp = line.split()
        if data_temp[0] = ID:
            time_real.append(datetime.strptime(line[19:32].strip(), "%m %d   %H  %M"))
            coor.append((data_temp[7],data_temp[8]))
    time_index = calc_time_nearest(datetime.strptime(time, "%Y-%m-%d %H:%M"), time_real)
    
    
time_cal = input_with_default('the time you want to calculate', '2013-11-21 11:33')
ID = input_with_default('drifter ID', 130410702)
lat_fcasted = input_with_default('latitude forecasted', 4150.1086)
lont_fcasted = input_with_default('longitude forecasted', 7005.7876)
datafile = 'http://www.nefsc.noaa.gov/drifter/drift_audubon_2013_1.dat'
f = open(datafile,'a+')
coor = pos_real(f, time_cal, ID)
distance = dist(coor[0], coor[1], lat_fcasted, lont_fcasted)