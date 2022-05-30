#!/usr/bin/env python3

"""
This code plots Lukas kande rain nowcast
written by IMS: 02/01/2022
for questions please contact Elyakom Vadislavsky, IMS R&D,
email: vadislavskye@ims.gov.il
"""

import matplotlib
matplotlib.use('Agg') # This is for runing without gui - unmark using plot show()
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl 
from mpl_toolkits.basemap import Basemap, addcyclic
import scipy.io
import numpy as np 
import numpy.ma as ma
import pandas as pd
import os
import sys
import glob
import time
import datetime
import calendar
import multiprocessing  
import pyproj
from PIL import Image

def readbilFC(filename,bilfactor):

	global NXinca,NYinca

	PSI = np.fromfile(filename,dtype='int16') # read binary file 'int16' --> INCA's short int
	l1 = len(PSI)
	l2 = int(l1/(NYinca*NXinca))
	PSI = np.reshape(PSI,(l2,(NYinca*NXinca))) # reshape array
	PSI = PSI/bilfactor
	
	return PSI

def encode_data_generic(datain,LEVELS):

	# This function encode data in 0,1,2,3... 
	# according to levels inupt

	ll = len(LEVELS)

	if datain < LEVELS[0]:
		nclr = -1		

	for step in range(1,ll):
		if LEVELS[step-1] <= datain and datain <= LEVELS[step]:
			nclr = step-1

	if LEVELS[ll-1] < datain:
		nclr = ll

	return nclr	


def plotfigure(fstep):

	# define global data
	global cmRain,m,dt
	global Xinca,Yinca,Xborder,Yborder,Xshed,Yshed
	global PATHOUT,patharch 
	global DATA
	global levels
	global NXinca,NYinca

		
	# draw map			
	
	# define border line width
	blw = 0.5	

	my_dpi = 72
	fig = plt.figure(figsize=(600/my_dpi, 800/my_dpi), dpi=my_dpi)
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	
	ft = dt+datetime.timedelta(minutes=fstep*timestep)

	print(fstep,ft.strftime("%d/%m/%Y %H:%M UTC"))

	out = np.reshape(DATA[fstep,:],(NYinca,NXinca))

	nj,ni = np.shape(out)
	outc = np.zeros((nj,ni))

	for j in range(nj):
		for i in range(ni):
			outc[j,i] = encode_data_generic(out[j,i],levels)

	VMIN = 0 # after encoding rain value begins from 1 ("0" is white color)
	VMAX = 14 # after encoding rain value end after 14 
	pm = m.pcolor(Xinca,Yinca,outc,vmin=VMIN,vmax=VMAX)
	pm.set_cmap(cmRain) # change to user colorbar
	pm.cmap.set_over([80./255.,0.,80./255.])
	pm.cmap.set_under('white')
      	
	cbar = plt.colorbar(pm,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],extend='both')  
	cbar.set_ticklabels(levels)
	cbar.set_label('mm/hr',fontsize=16)

	# draw Israels border	
	m.plot(Xborder,Yborder,linewidth=blw,color='k') # plot borders
	
	# draw coastlines 
	m.drawcoastlines(linewidth=blw,color='k')


	# draw parallers and meridians
	# Latitude
	parallels = np.arange(np.floor(leftLat),np.ceil(rightLat)+0.5,0.5)
	m.drawparallels(parallels,labels=[1,0,0,1],fontsize=12,color='k',linewidth=0.2)  
	# Longitude
	meridians = np.arange(np.floor(leftLon),np.ceil(rightLon)+0.5,0.5)
	m.drawmeridians(meridians,labels=[1,0,0,1],fontsize=12,color='k',linewidth=0.2) 
	
	xtxt,ytxt = m(leftLon,leftLat)
	plt.text(xtxt,ytxt,'Analysis: %s' % dt.strftime("%d/%m/%Y %H:%M UTC") , fontsize=12)

	plt.suptitle('LK extrap` %s\nFSTEP = +%03d [minutes]' % (ft.strftime("%d/%m/%Y %H:%M UTC"),fstep*timestep),fontsize=20)

	#plt.show()
	fout = '%s/rain_fc%02d.png' % (PATHOUT,fstep+1)
	plt.savefig(fout,dpi=my_dpi)	
	
	return 0
	
	

# main program #

if __name__ == "__main__":

	start = time.time()

	dt = datetime.datetime(2018,4,26,12,0)

	convert_to_mmhr = 0

	n_leadtimes = 25 # 2 hours extrapolation (every 5 minutes)
	timestep = 5
	
	if(convert_to_mmhr):
		n_leadtimes = 13 # 2 hours extrapolation (every 10 minutes)
		timestep = 10
	

	# Directory
	BILARCHIVE = '../LK_FORECAST'
	PATHOUT = '../GIF_LK_%dmin' % timestep 
	os.system('mkdir -p %s' % PATHOUT)
		
	# get map grid coords
	lola = np.loadtxt('../Config/inca_lonlat.asc')
	Lon = lola[:,::2] # get Longitude
	Lat = lola[:,1::2] # get Latitude

	leftLon = Lon.min()
	leftLat = Lat.min()
	rightLon =  Lon.max()
	rightLat = Lat.max()

	NYinca,NXinca = np.shape(Lat)
	
	print("NXinca: ",NXinca)
	print("NYinca: ",NYinca)
	
	# load color map
	H = scipy.io.loadmat('../Config/dbzbar.mat')
	cmR = H['dbzbar']
	ll = len(cmR)
	cmRain = cmR[1:ll-1,:]
	cmRain = mpl.colors.ListedColormap(cmRain)

	# precipitation levels:
	levels  = [0.05,0.1,0.2,0.7,1.2,2.,4.,6.,9.,13.,18.,24.,30.,50.,100.]
	f2read = '%s/%s/IMS_RADAR_PYSTEPS_LK_forecast_%dmin_steps_maps_%s.bil' % (BILARCHIVE,dt.strftime('%Y/%m/%d'),timestep,dt.strftime("%Y%m%d%H%M")) 
	print(f2read)
	DATA = readbilFC(f2read,100.)
	
	# create basic mapping data
	# define map
	m = Basemap(llcrnrlon=leftLon,llcrnrlat=leftLat,urcrnrlon=rightLon,urcrnrlat=rightLat,projection='mill',resolution='h')
	Xinca, Yinca = m(Lon, Lat) # get x & y from the source map

	# get Borders coordinates
	# Israel borders data Base
	D = scipy.io.loadmat('../Config/borders_data_base.mat')
	borders = D['borders']
	Xborder,Yborder = m(borders[:,0],borders[:,1]) # get x & y from the politic borders data base
	
	# serial mode
	#plotfigure(0)
	
	#### parrallel computing 
	pool = multiprocessing.Pool(9)
	pool.map(plotfigure, range(len(DATA))) 
	pool.close()
	pool.join()   
	print('done')
	####	
	

	end = time.time()
	print("Program finished after: %2.1f [minutes]" % ((end-start)/60.))

	
