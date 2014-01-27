import matplotlib.pyplot as plt
from mol_toolkits.basemap import Basemap
class add_figure(plt.figure):
    def __init__(self, lon, lat, interval_lon=1, interval_lon=1):
        '''
        draw the Basemap
        '''
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        cid = fig.canvas.mpl_connect('button_press_event', on_left_button_down)
        self.dmap = Basemap(projection='cyl',llcrnrlat=min(latsize)-0.01,urcrnrlat=max(latsize)+0.01,llcrnrlon=min(lonsize)-0.01,urcrnrlon=max(lonsize)+0.01,resolution='h')
        self.dmap.drawparallels(np.arange(int(min(latsize)),int(max(latsize))+1,interval_lat),labels=[1,0,0,0])
        self.dmap.drawmeridians(np.arange(int(min(lonsize))-1,int(max(lonsize))+1,interval_lon),labels=[0,0,0,1])
        self.dmap.drawcoastlines()
        self.dmap.fillcontinents(color='grey')
        self.dmap.drawmapboundary()
    def on_left_button_down(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            print 'You clicked: ', x, y
