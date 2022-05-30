#!/usr/bin/env python3

"""
This code calculates IMS radar lon/lat gridpoints
written by IMS: 02/01/2022
for questions please contact Elyakom Vadislavsky, IMS R&D,
email: vadislavskye@ims.gov.il
"""

import time
import numpy as np
from pyproj import Transformer
from pyproj import CRS
import matplotlib.pyplot as plt

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


def plotfigure(data):

	plt.figure()
	plt.imshow(data)
	plt.colorbar()
	plt.show()

	return 0

if __name__ == "__main__":

	start = time.time()

	# set global variables
	NX = NY = 561
	res = 1. # 1km
	RADlon,RADlat = calc_radar_lonlat(NX,NY,res)

	plotfigure(RADlon)
	plotfigure(RADlat)

	np.savetxt('../Config/RADlon.asc.gz',RADlon,fmt='%2.4f')
	np.savetxt('../Config/RADlat.asc.gz',RADlat,fmt='%2.4f')

	end = time.time()
	print('PROGRAM finished after %2.1f minutes' % ((end - start)/60.))




