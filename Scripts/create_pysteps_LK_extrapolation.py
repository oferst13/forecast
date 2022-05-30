#!/usr/bin/env python3

"""
This code calculates Lukas kande rain nowcast
written by IMS: 02/01/2022
for questions please contact Elyakom Vadislavsky, IMS R&D,
email: vadislavskye@ims.gov.il
"""

import os
import sys
import time
import datetime
import numpy as np
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from pysteps import motion, nowcasts
from pysteps.utils import transformation
import multiprocessing  
from pyproj import Transformer

def plotFigure(PSI):
	
	plt.figure()
	plt.imshow(PSI)
	plt.colorbar()
	plt.show()

	return 0

def write2bil(f2write,field,bil_factor):

	global PATHOUT

	print('Writing into %s' % f2write)

	f = open(f2write,'wb')
	
	x = np.int16(np.round(field*bil_factor)) 
	x.tofile(f)
	f.close()

	return 0 

@jit(nopython=True)
def regrid2inca(PSI):

	global Xrad,Yrad
	global NXradar,NYradar
	global NXinca,NYinca

	# regrid to INCA

	PSIout = np.zeros((NYinca,NXinca))
		 
	for j in range(NYinca):
		for i in range(NXinca): 
			if (0<=Xrad[j,i] and Xrad[j,i]<NXradar) and (0<= Yrad[j,i] and Yrad[j,i]<NYradar):
				PSIout[j,i] = PSI[Yrad[j,i],Xrad[j,i]]

	return PSIout

def radar2INCA(dt):

	global RADARDATADIR,NXinca,NXinca
	global QAMODE
	
	Rin = []
	out = []
	
	if (convert_to_mmhr):
		if (QAMODE):
			print("Reading INCA analysis maps")
			
		f2read = '%s/%s/RM_level85_%s.asc.gz' % (INCAMAPSDIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))		

	else:
		if (QAMODE):
			print("Reading IMS radar level 8 product")
			
		f2read = '%s/%s/IMS_radar_level_8_product_%s.asc.gz' % (RADARDATADIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))
	
	if os.path.isfile(f2read) and  os.stat(f2read).st_size != 0:
			
		if (QAMODE):
			print(f2read)
	
		# rain from level 8.5 product is in mm/hr

		Rin = pd.DataFrame.to_numpy(pd.read_csv(f2read, sep=' ', header=None,dtype='float')) 
		Rin[np.isnan(Rin)] = 0.
		Rin[Rin<rrth] = 0.
	
	
	if len(Rin)>0:	
	
	
		out = regrid2inca(Rin)		

	
	return out

def read_analysis_maps(dt):

	global INCA_ARCHIVE,hhfc,NYinca,NXinca,missval
	global QAMODE, n_leadtimes, timestep	
		
	nsteps = 3
	cnt = 0

	out = np.zeros((nsteps,NYinca,NXinca))

	for fstep in range (nsteps):

		ft = dt - datetime.timedelta(minutes=fstep*timestep)
			
		out[(nsteps-1)-fstep,:,:] = radar2INCA(ft)

		print((nsteps-1)-fstep)

		cnt+=1

	if cnt==3:

		return out

	else:

		return []

def create_RADAR_forecast(date):

	global datelist,n_leadtimes,PATHOUT

	
	R = read_analysis_maps(date)

	if len(R)>0:

		print("Proccessing %s" % date.strftime("%d/%m/%Y %H:%M"))

		if(convert_to_mmhr):
			R = R*6. # convert to mm/hr

		# https://pysteps.readthedocs.io/en/latest/auto_examples/plot_extrapolation_nowcast.html

		# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
		# set the fill value to -15 dBR
		R, _ = transformation.dB_transform(R, None, threshold=0.1, zerovalue=-15.0)
	
		# Estimate the motion field with Lucas-Kanade
		oflow_method = motion.get_method("LK")
		#V = oflow_method(R[-3:, :, :])
		V = oflow_method(R)		
		
		# Extrapolate the last radar observation
		extrapolate = nowcasts.get_method("extrapolation")
		
		R[~np.isfinite(R)] = -15. #metadata["zerovalue"]
		#R_f = extrapolate(R[-1, :, :], V, n_leadtimes)
		R_f = extrapolate(R[2, :, :], V, n_leadtimes)
	
		# Back-transform to rain rate
		R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]
		
		path1 = '%s/%s' % (PATHOUT,date.strftime('%Y/%m/%d'))
		os.system('mkdir -p %s' % path1)		
		f2write = '%s/IMS_RADAR_PYSTEPS_LK_forecast_%dmin_steps_maps_%s.bil' % (path1,timestep,date.strftime('%Y%m%d%H%M'))
		write2bil(f2write,R_f,100.)
		
	return 0	

if __name__ == "__main__":

	start = time.time()

	# define global constants
	convert_to_mmhr = 1

	n_leadtimes = 25 # 2 hours extrapolation (every 5 minutes)
	timestep = 5
	
	if(convert_to_mmhr):
		n_leadtimes = 13 # 2 hours extrapolation (every 10 minutes)
		timestep = 10
	
	rrth = 0.05
	QAMODE = 1
	
	PATHOUT = '../LK_FORECAST'
	RADARDATADIR = '../RADAR_PRODUCT_LEVEL_8'
	INCAMAPSDIR = '../RAINMAPS_LEVEL_85'	
	NXradar = NYradar = 561

	##################################################

	ITM = "epsg:2039" # define ITM coords
	wgs84 = "epsg:4326" # define wgs84 coords
	
	center_lat = 32.007 # IMS radar location latitude
	center_lon = 34.81456004 # IMS radar location longitude

	# how to tranform coordinates
	# https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1

	transformer = Transformer.from_crs(wgs84, ITM)
	Xcenter,Ycenter = transformer.transform(center_lat,center_lon) # transform between coords
	
	# get map grid coords
	lola = np.loadtxt('../Config/inca_lonlat.asc')
	INCAlon = lola[:,::2] # get Longitude
	INCAlat = lola[:,1::2] # get Latitude
	
	Xinca,Yinca = transformer.transform(INCAlat,INCAlon) # transform between coords
	
	# X and Y are in ITM (Israel Transverse mercator)
	# http://spatialreference.org/ref/epsg/2039/

	# Calculate RADAR's grid point location

	dx_from_center = (280000.+(Xinca-Xcenter))/1000. # devide by 1000 convert to meters
	dy_from_center = (280000.-(Yinca-Ycenter))/1000. # devide by 1000 convert to meters

	Xrad = (np.round(dx_from_center)).astype('int')  
	Yrad = (np.round(dy_from_center)).astype('int') 

	NYinca,NXinca = np.shape(INCAlon)

	#############################

	analysis_date = datetime.datetime(2018,4,26,12,0)
	create_RADAR_forecast(analysis_date)


	end = time.time()
	print("Program finished after: %2.1f [minutes]" % ((end-start)/60.))



	
