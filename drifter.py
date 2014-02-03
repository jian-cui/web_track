from moj import jata
from datetime import datetime
import readline
from matplotlib import pyplot as plt
def data_extracted(filename,drifter_id=None,starttime=None):
    data = {}
    f = open(filename, 'r')
    d_id, d_time, d_lon, d_lat = [], [], [], []
    for line in readline(f):
        line = line.split()
        did.append(line[0])
        dtime.append(datetime(year=2013,month=int(line[2]),day=int(line[3]),
                              hour=int(line[4]),minute=int(line[5])))
        dlon.append(line[7])
        dlat.append(line[8])
    if drifter_id is not None:
        i = indexofdata(d_id, drifter_id)
        if starttime is not None:
            dtime_temp = d_time[i[0]:i[-1]+1]
            j = indexofdata(d_time_temp, starttime)
            data['time'] = dtime[i[0]:i[-1]+1][j:]
            data['lon'] = dlon[i[0]:i[-1]+1][j:]
            data['lat'] = dlat[i[0]:i[-1]+1][j:]
        else:
            data['time'] = dtime[i[0]:i[-1]+1]
            data['lon'] = dlon[i[0]:i[-1]+1]
            data['lat'] = dlat[i[0]:i[-1]+1]
    elif drifter_id is None and starttime is None:
        data['time'] = dtime
        data['lon'] = dlon
        data['lon'] = dlat
    return data
def indexofdata(dlist,data):
    index = []
    startindex = dlist.index(data)
    j = startindex
    for i in dlist[startindex:]:
        if i == data:
            index.append(j)
        j+=1
        # if i != data:
            # break
    return index

starttime = datetime(year=2013, month=9, day=29, hour=11, minute=46)
drifter_ID = jata.input_with_default('drifter ID', 139420691)
filename = "/net/home3/ocn/jmanning/py/jc/web_track/drift_tcs_2013_1.dat"

data = data_extracted(filename, drifter_ID, starttime)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
