#!/usr/bin/env python3

"""
This code calculates STEPS 50 member ensemble rain nowcast
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
from pyproj import Transformer
from numba import jit

from pprint import pprint
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing import ensemblestats
from pysteps.utils import conversion, dimension, transformation


def plotFigure(PSI):
	
	plt.figure()
	plt.imshow(PSI)
	plt.colorbar()
	plt.show()

	return 0




###############################################

@jit(nopython=True)
def RADAR2INCA_grid(PSI):

	global NXradar,NYradar,NYinca,NXinca,Xinca_itm,Yinca_itm
	
	# takes nearest grid point

	out = np.zeros((NYinca,NXinca))

	for j in range(NYinca):
		for i in range(NXinca): 

			Xp = Xinca_itm[j,i] 
			Yp = Yinca_itm[j,i]

			# X and Y are in ITM (Israel Transverse mercator)
			# http://spatialreference.org/ref/epsg/2039/

			# Calculate RADAR's grid point location
		
			dx_from_center = (280000.+(Xp-Xcenter))/1000. # devide by 1000 convert to meters
			dy_from_center = (280000.-(Yp-Ycenter))/1000. # devide by 1000 convert to meters

			Xrad = int(round(dx_from_center))  
			Yrad = int(round(dy_from_center)) 

			if 0<=Xrad and Xrad<NXradar and 0<= Yrad and Yrad<NYradar:
				out[j,i] = PSI[Yrad,Xrad]
				
	return out

def read_ims_radar(dt):

	global NXinca, NYinca, box_minimum

	# take only 3 steps for extrapolation, need "-1" and "-3"

	nsteps = 3
	cnt = 0

	out = np.zeros((nsteps,NYinca,NXinca))

	for fstep in range(nsteps):

		ft = dt - datetime.timedelta(minutes=fstep*5)

		f2read = '%s/%s/IMS_radar_level_8_product_%s.asc.gz' % (RADARDATADIR,ft.strftime('%Y/%m/%d'),ft.strftime('%Y%m%d%H%M'))
		
		if os.path.isfile(f2read) and os.stat(f2read).st_size != 0:

			# get radar sweep data in mm/hr
			print("Reading:", f2read)

			rain_arr = pd.DataFrame.to_numpy(pd.read_csv(f2read, sep=' ', header=None)) 
	
			rain_arr[np.isnan(rain_arr)] = 0. # fill nan values with 0.
	
			radar = RADAR2INCA_grid(rain_arr)

			print((nsteps-1)-fstep)

			# now make sure we have at least 1% of data as radar echo

			out_array = np.zeros(np.shape(radar))
	
			# 0.1 mm/hr is minimum for radar signal

			out_array[radar>0.1] = 1
			out_count = np.nansum(out_array)

			if out_count/float(NYinca*NXinca)>= box_minimum:
				out[(nsteps-1)-fstep,:,:] = radar
				cnt+=1

	if cnt==3:

		return out

	else:

		return []

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
			print("Reading INCA analysis maps units[mm/10min]")
			
		f2read = '%s/%s/RM_level85_%s.asc.gz' % (INCAMAPSDIR,dt.strftime('%Y/%m/%d'),dt.strftime('%Y%m%d%H%M'))		

	else:
		if (QAMODE):
			print("Reading IMS radar level 8 product units[mm/hr]")
			
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
			
		radar = radar2INCA(ft)

		print((nsteps-1)-fstep)

		# now make sure we have at least 1% of data as radar echo

		out_array = np.zeros(np.shape(radar))

		# 0.1 mm/hr is minimum for radar signal

		out_array[radar>0.1] = 1
		out_count = np.nansum(out_array)

		if out_count/float(NYinca*NXinca)>= box_minimum:
			out[(nsteps-1)-fstep,:,:] = radar
			cnt+=1

	if cnt==3:

		return out

	else:

		return []



def create_RADAR_ens_forecast(date):

	global datelist,n_leadtimes,PATHOUT

	print("\n\n\n***************\n\n\nProcessing %s" % date.strftime('%d/%m/%Y %H:%M UTC'))
	
	R = read_analysis_maps(date)	

	if len(R)>0:

		if(convert_to_mmhr):
			R = R*6. # convert to mm/hr

		R = R[:,1:,1:] # reduce INCA grid by 1 by 1 pixel to make it even !!!
		
		# https://pysteps.readthedocs.io/en/latest/auto_examples/plot_ensemble_verification.html

		metadata_base = dict(xpixelsize=1000, ypixelsize=1000,unit='mm/h')

		#The data are upscaled to 2 km resolution to limit the memory usage and thus be able to afford a larger number of ensemble members.
							
		# Upscale data to 2 km
		R, metadata = dimension.aggregate_fields_space(R, metadata_base, 2000)
		
		# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
		# set the fill value to -15 dBR
		R, _ = transformation.dB_transform(R, None, threshold=0.1, zerovalue=-15.0)
	
		# Set missing values with the fill value
		R[~np.isfinite(R)] = -15.0

		# Nicely print the metadata
		#pprint(metadata)


		###############################################################################
		# Forecast
		# --------
		#
		# We use the STEPS approach to produce a ensemble nowcast of precipitation fields.

		# Estimate the motion field
		V = dense_lucaskanade(R)

		# Perform the ensemble nowcast with STEPS
		nowcast_method = nowcasts.get_method("steps")
		R_f = nowcast_method(
		    R[-3:, :, :],
		    V,
		    n_leadtimes,
		    n_ens_members,
		    n_cascade_levels=6,
		    R_thr=-10.0,
		    kmperpixel=2.0,
		    timestep=timestep,
		    decomp_method="fft",
		    bandpass_filter_method="gaussian",
		    noise_method="nonparametric",
		    vel_pert_method="bps",
		    mask_method="incremental",
		    seed=seed,
		    num_workers=num_workers,	
		)

		# Back-transform to rain rates
		R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]
		R_f_ = np.zeros(R_f.shape)
		R_f_[:,:] = R_f

		R_f = R_f.astype("single") #  # single precision to improve computational efficiency

		path1 = '%s/%s' % (PATHOUT,date.strftime('%Y/%m/%d'))
		os.system('mkdir -p %s' % path1)
		fout = '%s/STEPS_%d_mem_ens_forecast_%dmin_steps_maps_%s.npz' % (path1,n_ens_members,timestep,date.strftime('%Y%m%d%H%M'))
		np.savez_compressed(fout,R_f = R_f)

		
		
		
	
	return 0	

if __name__ == "__main__":

	start = time.time()

	# define global constants
	convert_to_mmhr = 0

	n_leadtimes = 24 
	timestep = 5

	if(convert_to_mmhr): 
		n_leadtimes = 12 
		timestep = 10

	n_ens_members = 50
	seed = 24
	num_workers = 10
	missval = -999.
	QAMODE = 1
	rrth = 0.05

	box_minimum = 0.01 # minimum size for area with thresh hold out (0.01 is 1%)
	
	PATHOUT = '../STEPS_FORECAST' 
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
	
	#############
	# go serial #	
	
	analysis_date = datetime.datetime(2018,4,26,12,0)
	create_RADAR_ens_forecast(analysis_date)
		
	
	end = time.time()
	print("Program finished after: %2.1f [minutes]" % ((end-start)/60.))



	
