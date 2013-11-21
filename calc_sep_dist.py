'''
created on Nov 21, 11:20
compare data forecasted with data recorded by drifter.
'''
import xlrd
import xlwt
from conversions import dist

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
def update_default_date(data, v_default):
    value = v_defult
    value = input_with_defalut(data, value)    
    return value
time_cal = input_with_default('the time you want to calculate', '2013-11-21 11:33')
ID = input_with_default('drifter ID', 130410702)
lat_fcasted = input_with_default('latitude forecasted', 4150.1086)
lont_fcasted = input_with_default('longitude forecasted', 7005.7876)
datafile = 'http://www.nefsc.noaa.gov/drifter/drift_audubon_2013_1.dat'
f = open(datafile,'a+')

