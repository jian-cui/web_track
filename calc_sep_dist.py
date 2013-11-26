'''
created on Nov 21, 11:20
compare data forecasted with data recorded by drifter.
'''
import xlrd
import xlwt
from conversions import dist
from datetime import datetime
import time
#from scipy import interpolate
#import numpy as np
#from pydap.client import open_url
from urllib import urlopen  

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

def calc_time_nearest(time_want_to_cal, time_real):
    t = time.mktime(time_want_to_cal)
    for i in time_real:
        j = time.mktime(i)
        if t > j:
            continue
        elif t == j:
            return time_real.index(i)
            break
        else:
            return time_real.index(i)-1
            break

def pos_real(f, time_cal, ID):
    time_real = []
    coor = []
    for line in f.readlines():
        data_temp = line.split()
        if data_temp[0] == ID:
            time_real.append(time.strptime(line[19:32].strip(), "%m %d   %H  %M"))
            coor.append((float(data_temp[7]), float(data_temp[8])))
        else:
            pass
    index = calc_time_nearest(time_cal, time_real)
    time_f = time.mktime(time_cal)
#    x = np.arange(coor[index][0], coor[index+1][0])
#    y = np.arange(coor[index][1], coor[index+1][1])
    x_cal = (time_f - time.mktime(time_real[index])) * (coor[index+1][0] - coor[index][0]) / (time.mktime(time_real[index+1]) - time.mktime(time_real[index])) + coor[index][0]
    y_cal = (time_f - time.mktime(time_real[index])) * (coor[index+1][1] - coor[index][1]) / (time.mktime(time_real[index+1]) - time.mktime(time_real[index])) + coor[index][1]
#    f = interpolate.
#    y_cal = f(np.arange(x_cal))
    return (x_cal, y_cal)

#def save_excel(filename, data, head = ('date_cal', 'ID', 'distance')):
#    dexcel = xlrd.open_workbook(filename)
#    dtable = dexcel.sheets()[0]
#    rows = dtable.nrows
##    cols = dtable.ncols
#    if not rows:
#        for i in head:
#            dtable.put_cell(rows, head.index(i), 1, i, 0)
#            dtable.put_cell(rows+1, head.index(i), 1, data[head.index(i)], 0)
#    else:
#        for i in data:
#            dtable.put_cell(rows, data.index(i), 1, data[data.index(i)], 0)
def write_line(f, data):
    line = str(data[0])
    for word in data[1:]:
        line = line + ' ' + str(word)
    f.write(line + '\n')
    
def write_data(f, data, head = ''):
    if not len(f.read()):
        f.write(head + '\n')
    f.seek(0, 2)
    write_line(f, data)

#def save_excel1(filename, data, head):
#    dexcel = xlwt.Workbook()
#    dtable = dexcel.sheets()[0]
#    rows = dtable.nrows
#    if not rows:
#        for i in head:
#            dtable.put_cell(rows, head.index(i), i)
#            dtable.put_cell(rows+1, head.index(i), data[head.index(i)])
#    else:
#        for i in data:
#            dtable.put_cell(rows, data.index(i), data[data.index(i)])
#    dexcel.save
time_input = input_with_default('the time you want to calculate', '11-21 11:33')
time_cal = time.strptime(time_input, "%m-%d %H:%M")
ID = input_with_default('drifter ID', 130410701)
lat_fcasted = input_with_default('latitude forecasted', 4150.1086)
lont_fcasted = input_with_default('longitude forecasted', 7005.7876)
datafile = r'http://www.nefsc.noaa.gov/drifter/drift_audubon_2013_1.dat'
f = urlopen(datafile)
#f = open(datafile)
coor = pos_real(f, time_cal, ID)
distance = dist(coor[0], coor[1], float(lat_fcasted), float(lont_fcasted))
print distance[0]
data = (time_input, ID, str(distance[0]))
#save_excel('distance.xls', data = (time_input, ID, str(distance[0])))
fs = open('distance.dat', 'a+')
write_data(fs, data, 'date ID distance')
fs.close()