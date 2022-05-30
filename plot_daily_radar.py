#!/usr/bin/env python3

"""
This code plot daily accumulated rain
written by IMS: 02/01/2022
for questions please contact Elyakom Vadislavsky, IMS R&D,
email: vadislavskye@ims.gov.il
"""

import time
import datetime
import sys
import os
import numpy as np
import pandas as pd
import scipy.io
from pyproj import Transformer
from pyproj import CRS
import matplotlib
matplotlib.use('Agg') # This is for runing without gui - unmark using plot show()
import matplotlib.pyplot as plt
import matplotlib as mpl 
from mpl_toolkits.basemap import Basemap



###############################################################
#  graphics functions
def encode_data_genericRR(datain,levels):

	# This function encode data in 0,1,2,3... 
	# according to levels inupt


	ll = len(levels)

	if datain < levels[0]:
		nclr = -1		

	for step in range(1,ll):
		if levels[step-1] <= datain and datain < levels[step]:
			nclr = step-1


	if levels[ll-1] <= datain:
		nclr = ll

	return nclr

def calc_radar_lonlat(NX,NY,res):

	wgs84 = "epsg:4326" # define wgs84 coords
	ITM = "epsg:2039" # define ITM coords
 
	# IMS RADAR location in ITM coordinates 
	lat0 = 32.007
	lon0 = 34.81456004

	transformerITM = Transformer.from_crs(wgs84, ITM)
	Xcenter,Ycenter = transformerITM.transform(lat0,lon0) # transform between coords

	nx = (NX-1)/(2/(1./res))
	ny = (NY-1)/(2/(1./res))

	x = np.linspace(-nx, nx, NX)
	y = np.linspace(ny, -ny, NY)
	
	xv, yv = np.meshgrid(x, y)

	xv = xv*1000.
	yv = yv*1000.

	RADX = Xcenter+xv
	RADY = Ycenter+yv

	# Convert from LAT, LON to ITM (Israel Transverse Mercator) X, Y
	transformerITM_back = Transformer.from_crs(ITM,wgs84)
	RADlat,RADlon = transformerITM_back.transform(RADX,RADY)

	return RADlon,RADlat

#################################################################

# plot Image 

def plotImage(dt,fout,PSI1,PSI2,PSI3):

	global Xradar,Yradar,Xborders,Yborders
	global m,cmRain

	####################
	# load data for plot

	# draw data			

	my_dpi = 100
	fig = plt.figure(figsize=(1200/my_dpi, 1000/my_dpi), dpi=my_dpi)
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	
	# precipitation levels:
	levels = [0.05,0.1,1.,4.,7.,10.,15.,20.,25.,30.,40.,50.,60.,80.,100.] 

	# define border line width
	blw = 0.5	


	############################################

	# Draw RADAR data

	plt.subplot(1,3,1)

	out = PSI1

	nj,ni = np.shape(out)

	for j in range(0,nj):
		for i in range(0,ni):
			out[j,i] = encode_data_genericRR(out[j,i],levels)

	pm = m.pcolor(Xradar,Yradar,out,vmin=0,vmax=14)
	pm.set_cmap(cmRain) # change to user colorbar

	# draw Israels border	
	m.plot(Xborders,Yborders,linewidth=blw,color='k') # plot borders
	
	# draw coastlines 
	m.drawcoastlines(linewidth=blw)
	
	# draw parallers and meridians
	# Latitude
	parallels = np.arange(29.5,34.,0.5)
	m.drawparallels(parallels,labels=[1,0,0,1],fontsize=10,color='w')  
	# Longitude
	meridians = np.arange(34.,36.5,0.5)
	m.drawmeridians(meridians,labels=[1,0,0,1],fontsize=10,color='w') 

	str1 = 'RR' 
	plt.title(str1,fontsize=16)
	
	############################################
	
	# Draw IDW data

	plt.subplot(1,3,2)

	out = PSI2

	nj,ni = np.shape(out)

	for j in range(0,nj):
		for i in range(0,ni):
			out[j,i] = encode_data_genericRR(out[j,i],levels)

	pm = m.pcolor(Xradar,Yradar,out,vmin=0,vmax=14)
	pm.set_cmap(cmRain) # change to user colorbar

	# draw Israels border	
	m.plot(Xborders,Yborders,linewidth=blw,color='k') # plot borders

	# draw coastlines 
	m.drawcoastlines(linewidth=blw)

	# draw parallers and meridians
	# Latitude
	parallels = np.arange(29.5,34.,0.5)
	m.drawparallels(parallels,labels=[1,0,0,1],fontsize=10,color='w')  
	# Longitude
	meridians = np.arange(34.,36.5,0.5)
	m.drawmeridians(meridians,labels=[1,0,0,1],fontsize=10,color='w') 	

	str1 = 'PA' 
	plt.title(str1,fontsize=16)

	############################################

	# Draw RADAR corrected data

	plt.subplot(1,3,3)

	out = PSI3

	nj,ni = np.shape(out)

	for j in range(0,nj):
		for i in range(0,ni):
			out[j,i] = encode_data_genericRR(out[j,i],levels)

	pm = m.pcolor(Xradar,Yradar,out,vmin=0,vmax=14)
	pm.set_cmap(cmRain) # change to user colorbar

	# draw Israels border	
	m.plot(Xborders,Yborders,linewidth=blw,color='k') # plot borders

	# draw coastlines 
	m.drawcoastlines(linewidth=blw)
	
	# draw parallers and meridians
	# Latitude
	parallels = np.arange(29.5,34.,0.5)
	m.drawparallels(parallels,labels=[1,0,0,1],fontsize=10,color='w')  
	# Longitude
	meridians = np.arange(34.,36.5,0.5)
	m.drawmeridians(meridians,labels=[1,0,0,1],fontsize=10,color='w') 

	str1 = 'RM'
	plt.title(str1,fontsize=16)

	############################################
	
	############################################

	
	cbaxes = fig.add_axes([0.14, 0.1, 0.76, 0.03]) # add_axes refer to [left, bottom, width, height]
	cbar = plt.colorbar(pm,orientation="horizontal",ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],extend='both',cax = cbaxes)  
	cbar.cmap.set_over([80./255.,0.,80./255.])
	cbar.cmap.set_under('white')
	#cbar.set_ticklabels([0.1,1,3,5,7,10,15,20,25,30,40,50,60,80,100])
	cbar.set_ticklabels(levels)
	cbar.set_label('mm',fontsize=16)
	
	
	# add title and save file

	title1 = 'Daily (24H) Precipitation Analysis %s UTC [mm]' % (dt.strftime('%d/%m/%Y %H:%M'))

	plt.suptitle(title1,fontsize=18)
	plt.savefig(fout,dpi=my_dpi)
	plt.close()


	#plt.show()

	return 0
	############################################



def procdata(dt):

	global DATADIR,PATHOUT
		
	# read the data
	RR = [] 
	PA = [] 
	RM = [] 
	
	f1 = '%s/%s/RR24H_level85_%s.asc.gz' % (DATADIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))
	f2 = '%s/%s/PA24H_level85_%s.asc.gz' % (DATADIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))
	f3 = '%s/%s/RM24H_level85_%s.asc.gz' % (DATADIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))
	
	if os.path.isfile(f1) and os.stat(f1).st_size != 0:

		#RR = np.loadtxt(f1)
		RR = pd.DataFrame.to_numpy(pd.read_csv(f1, sep=' ', header=None))

	if os.path.isfile(f2) and os.stat(f2).st_size != 0:

		#PA = np.loadtxt(f2)
		PA = pd.DataFrame.to_numpy(pd.read_csv(f2, sep=' ', header=None))

	if os.path.isfile(f3) and os.stat(f3).st_size != 0:

		#RM = np.loadtxt(f3)
		RM = pd.DataFrame.to_numpy(pd.read_csv(f3, sep=' ', header=None))


	if len(RR)!=0. and len(PA)!=0. and len(RM)!=0.:
		
		fout = '%s/DailyAnalysis_%s.png' % (PATHOUT,dt.strftime('%Y%m%d%H%M'))
	
		plotImage(dt,fout,RR,PA,RM)
	
	return 0

########
# main #
########

if __name__ == "__main__":

	start = time.time() # start running time stopper

	# set global variables
	NX = NY = 561
	res = 1. # 1km
	RADlon,RADlat = calc_radar_lonlat(NX,NY,res)

	#######################################################################################################
	# create Basemap instance
	# for INCA grid
	m = Basemap(llcrnrlon=34.,llcrnrlat=29.3,urcrnrlon=36.,urcrnrlat=33.5,projection='mill',resolution='h')
	print("Map on INCA grid ready !!!")

	# get RADAR map grid coords
	Xradar, Yradar = m(RADlon, RADlat) # get x & y on INCA grid

	# get Borders coordinates
	# Israel borders data Base
	D = scipy.io.loadmat('../Config/borders_data_base.mat')
	borders = D['borders']
	Xborders,Yborders = m(borders[:,0],borders[:,1])

	######################################################

	# load color map
	H = scipy.io.loadmat('../Config/dbzbar.mat')
	cmR = H['dbzbar']
	ll = len(cmR)
	cm1 = cmR[1:ll-1,:]
	cm2 = cm1
	cmRain = mpl.colors.ListedColormap(cm1)


	##########################################################
	DATADIR = '../RAINMAPS_LEVEL_85' 
	PATHOUT = '../IMAGEOUT' 
	os.system('mkdir -p %s' % (PATHOUT))

	analysis_date = datetime.datetime(2018,4,26,6,0)
	
	procdata(analysis_date)

	end = time.time()
	print('PROGRAM finished after %2.1f minutes' % ((end - start)/60.))

