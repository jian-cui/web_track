import jata
from datetime import datetime
import readline
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import time

def data_extracted(filename,drifter_id=None,starttime=None):
    '''
    get a dict made of time, lon, lat from local file.
    filename: local file diretory
    drifter_id: the specific data of some id you want.
    starttime: have to be input with drifter_id, or just drifter_id.
    '''
    data = {}
    did, dtime, dlon, dlat = [], [], [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            try:
                line = line.split()
                did.append(int(line[0]))
                dtime.append(datetime(year=2013,
                                      month=int(line[2]),day=int(line[3]),
                                      hour=int(line[4]),minute=int(line[5])))
                dlon.append(float(line[7]))
                dlat.append(float(line[8]))
            except IndexError:
                continue
    if drifter_id is not None:
        i = index_of_value(did, drifter_id)
        if starttime is not None:
            dtime_temp = dtime[i[0]:i[-1]+1]
            j = index_of_value(dtime_temp, starttime)
            data['time'] = dtime[i[0]:i[-1]+1][j[0]:]
            data['lon'] = dlon[i[0]:i[-1]+1][j[0]:]
            data['lat'] = dlat[i[0]:i[-1]+1][j[0]:]
        else:
            data['time'] = dtime[i[0]:i[-1]+1]
            data['lon'] = dlon[i[0]:i[-1]+1]
            data['lat'] = dlat[i[0]:i[-1]+1]
    elif drifter_id is None and starttime is None:
        data['time'] = dtime
        data['lon'] = dlon
        data['lat'] = dlat
    else:
        raise ValueError("Please input drifter_id while starttime is input")
    # try:
    #     i = index_of_value(did, drifter_id)
    #     try:
    #         dtime_temp = dtime[i[0]:i[-1]+1]
    #         j = index_of_value(dtime_temp, starttime)
    #         data['time'] = dtime[i[0]:i[-1]+1][j[0]:]
    #         data['lon'] = dlon[i[0]:i[-1]+1][j[0]:]
    #         data['lat'] = dlat[i[0]:i[-1]+1][j[0]:]
    #     except ValueError:
    #         data['time'] = dtime[i[0]:i[-1]+1]
    #         data['lon'] = dlon[i[0]:i[-1]+1]
    #         data['lat'] = dlat[i[0]:i[-1]+1]
    # except ValueError:
    #     if starttime is None:
    #         data['time'] = dtime
    #         data['lon'] = dlon
    #         data['lat'] = dlat
    #     else:
    #         raise ValueError("Please input drifter_id while starttime is input")
    return data
def index_of_value(dlist,dvalue):
    '''
    return the indices of dlist that equals dvalue
    '''
    index = []
    startindex = dlist.index(dvalue)
    i = startindex
    for v in dlist[startindex:]:
        if v == dvalue:
            index.append(i)
        i+=1
    return index
      
starttime = datetime(year=2013, month=9, day=29, hour=11, minute=46)
drifter_id = jata.input_with_default('drifter ID', 139420691)
filename = "/net/home3/ocn/jmanning/py/jc/web_track/drift_tcs_2013_1.dat"
time1 =time.time()

data = data_extracted(filename, drifter_id, starttime)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
minlat, maxlat = min(data['lat']), max(data['lat'])
minlon, maxlon = min(data['lon']), max(data['lon'])
bsmap = Basemap(projection='cyl',
                llcrnrlat=minlat-1, urcrnrlat=maxlat+1,
                llcrnrlon=minlon-1, urcrnrlon=maxlon+1,
                resolution='h',ax=ax)
bsmap.drawparallels(range(int(minlat), int(maxlat)+1), labels=[1,0,0,0])
bsmap.drawmeridians(range(int(minlon), int(maxlon)+1), labels=[0,0,0,1])
bsmap.drawcoastlines()
bsmap.fillcontinents(color='gray')
bsmap.drawmapboundary()
ax.plot(data['lon'], data['lat'], 'ro-')
plt.show()
time2 = time.time()
print time2-time1
