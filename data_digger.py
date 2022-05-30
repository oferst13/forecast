#!/usr/bin/env python3

"""
This code calculates rainfall at chosen point
written by IMS: 02/01/2022
for questions please contact Elyakom Vadislavsky, IMS R&D,
email: vadislavskye@ims.gov.il
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj import CRS
import matplotlib.pyplot as plt


##################
#   functions    # 
##################

def get_current_rain(dt):

	global DATADIR

	
	RM = [] # case of no rain
	
	f1 = '%s/%s/RM24H_level85_%s.asc.gz' % (DATADIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))
	
	RM = []

	if os.path.isfile(f1) and os.stat(f1).st_size != 0:

		print("Reading",f1)
		#RM = np.loadtxt(f1)
		RM = pd.DataFrame.to_numpy(pd.read_csv(f1, sep=' ', header=None))

	return RM



def get_data(dt,LON,LAT):

	global missval,fout
	global wgs84,ITM
	global Xcenter,Ycenter
	
	# navigation section
	
	# Convert from LAT, LON to ITM (Israel Transverse Mercator) X, Y
	X,Y = transformerITM.transform(LAT,LON) # transform between coords	

	# X and Y are in ITM (Israel Transverse mercator)
	# http://spatialreference.org/ref/epsg/2039/

	# Calculate RADAR's grid point location
	Xrad = int(round(((280000.+(X-Xcenter))/1000.)/1.))  
	Yrad = int(round(((280000.-(Y-Ycenter))/1000.)/1.)) 

	###################################
	# Precipitation section
	
	# get current rain data 

	RM = get_current_rain(dt)

	Rcurrent = missval

	if len(RM)!=0:
		rdata1 = RM[Yrad,Xrad]
		if rdata1 >= rain10_threshold:	
			Rcurrent = rdata1

			print("At Lat: %2.4fN, Lon: %2.4f, the precipitation is: %2.1f" % (LAT,LON,Rcurrent))	

			plotfigure(RM,Xrad,Yrad)

	return 0


def plotfigure(DATA,x,y):

	DATA[DATA<0.1] = nan

	plt.figure()
	plt.imshow(DATA)
	plt.plot(x,y,'ro',markersize=4,markerfacecolor='none')
	plt.plot(x,y,'ro',markersize=8,markerfacecolor='none')
	plt.plot(x,y,'r+',markersize=8)
	str1 = '%2.1f' % DATA[y,x]
	plt.text(x,y,str1,fontsize=14,color='red')
	plt.colorbar()
	plt.show()

	return 0

###################################



########
# main #
########

if __name__ == "__main__":
	
	start = time.time()
	
	missval = -999.
	rain10_threshold = 0.05 # 0.05 mm/hr drops at 10 minutes interval

	DATADIR = '../RAINMAPS_LEVEL_85' 

	# Navigation projection defintion

	wgs84 = "epsg:4326" # define wgs84 coords
	ITM = "epsg:2039" # define ITM coords
 
	# IMS RADAR location in ITM coordinates 
	lat0 = 32.007
	lon0 = 34.81456004

	transformerITM = Transformer.from_crs(wgs84, ITM)
	Xcenter,Ycenter = transformerITM.transform(lat0,lon0) # transform between coords
	  
	analysis_date = datetime.datetime(2018,4,26,6,0)
	# run the function

	lat = 31.5 # North
	lon = 35.25 # East
	get_data(analysis_date,lon,lat) 


	end = time.time()
	print('PROGRAM finished after %2.1f minutes' % ((end - start)/60.))


