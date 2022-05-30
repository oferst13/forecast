#!/usr/bin/env python3

"""
This code plots STEPS 50 member ensemble rain nowcast
written by IMS: 02/01/2022
for questions please contact Elyakom Vadislavsky, IMS R&D,
email: vadislavskye@ims.gov.il
"""

import numpy as np
import numpy.ma as ma
import warnings
from scipy.ndimage import convolve
import scipy.io
import os
import sys
import time
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import multiprocessing 
from PIL import Image

###############################################

"""
pysteps.downscaling.rainfarm
============================
Implementation of the RainFARM stochastic downscaling method as described in
:cite:`Rebora2006`.
.. autosummary::
    :toctree: ../generated/
    downscale
"""




def _log_slope(log_k, log_power_spectrum):
    lk_min = log_k.min()
    lk_max = log_k.max()
    lk_range = lk_max - lk_min
    lk_min += (1 / 6) * lk_range
    lk_max -= (1 / 6) * lk_range

    selected = (lk_min <= log_k) & (log_k <= lk_max)
    lk_sel = log_k[selected]
    ps_sel = log_power_spectrum[selected]
    alpha = np.polyfit(lk_sel, ps_sel, 1)[0]
    alpha = -alpha

    return alpha


def _balanced_spatial_average(x, k):
    ones = np.ones_like(x)
    return convolve(x, k) / convolve(ones, k)


def downscale(precip, alpha=None, ds_factor=2, threshold=None, return_alpha=False):
    """
    Downscale a rainfall field by a given factor.
    Parameters
    ----------
    precip: array_like
        Array of shape (m,n) containing the input field.
        The input is expected to contain rain rate values.
    alpha: float, optional
        Spectral slope. If none, the slope is estimated from
        the input array.
    ds_factor: int, optional
        Downscaling factor.
    threshold: float, optional
        Set all values lower than the threshold to zero.
    return_alpha: bool, optional
        Whether to return the estimated spectral slope `alpha`.
    Returns
    -------
    r: array_like
        Array of shape (m*ds_factor,n*ds_factor) containing
        the downscaled field.
    alpha: float
        Returned only when `return_alpha=True`.
    Notes
    -----
    Currently, the pysteps implementation of RainFARM only covers spatial downscaling.
    That is, it can improve the spatial resolution of a rainfall field. However, unlike
    the original algorithm from Rebora et al. (2006), it cannot downscale the temporal
    dimension.
    References
    ----------
    :cite:`Rebora2006`
    """

    ki = np.fft.fftfreq(precip.shape[0])
    kj = np.fft.fftfreq(precip.shape[1])
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)

    ki_ds = np.fft.fftfreq(precip.shape[0] * ds_factor, d=1 / ds_factor)
    kj_ds = np.fft.fftfreq(precip.shape[1] * ds_factor, d=1 / ds_factor)
    k_ds_sqr = ki_ds[:, None] ** 2 + kj_ds[None, :] ** 2
    k_ds = np.sqrt(k_ds_sqr)

    if alpha is None:
        fp = np.fft.fft2(precip)
        fp_abs = abs(fp)
        log_power_spectrum = np.log(fp_abs ** 2)
        valid = (k != 0) & np.isfinite(log_power_spectrum)
        alpha = _log_slope(np.log(k[valid]), log_power_spectrum[valid])

    fg = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(*k_ds.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg *= np.sqrt(k_ds_sqr ** (-alpha / 2))
    fg[0, 0] = 0
    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)

    P_u = np.repeat(np.repeat(precip, ds_factor, axis=0), ds_factor, axis=1)
    rad = int(round(ds_factor / np.sqrt(np.pi)))
    (mx, my) = np.mgrid[-rad : rad + 0.01, -rad : rad + 0.01]
    tophat = ((mx ** 2 + my ** 2) <= rad ** 2).astype(float)
    tophat /= tophat.sum()

    P_agg = _balanced_spatial_average(P_u, tophat)
    r_agg = _balanced_spatial_average(r, tophat)
    r *= P_agg / r_agg

    if threshold is not None:
        r[r < threshold] = 0

    if return_alpha:
        return r, alpha

    return r


###############################################


def read_ensemble_forecast(dt):

	global n_ens_members, ENSEMBLE_ARCHIVE,timestep

	PSI = []

	fens = '%s/%s/STEPS_%d_mem_ens_forecast_%dmin_steps_maps_%s.npz' % (ENSEMBLE_ARCHIVE,dt.strftime('%Y/%m/%d'),n_ens_members,timestep,dt.strftime('%Y%m%d%H%M'))
	
	if os.path.isfile(fens) and os.stat(fens).st_size != 0:

		# read ensemble forecast file
		print("Reading:", fens)

		PSI = np.load(fens)['R_f']
	
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

def plotFigure(PSI):

	plt.figure()
	PSI[PSI<0.] = np.nan
	plt.imshow(PSI)
	plt.colorbar()
	plt.show()

	return 0

def plotFigure_ens_member(fstep):

	# define global data
	global Xinca,Yinca,Xborder,Yborder,timestep
	global R_o,date_analysis,PATHOUT,ens_mem

	
	PSIrain_ = downscale(R_o[ens_mem,fstep,:,:],alpha=None, ds_factor=2)		
	PSIrain = np.full((NYinca,NXinca),missval)
	PSIrain[1:,1:] = PSIrain_
	
	# draw map			
	
	# define border line width
	blw = 0.5	

	my_dpi = 100
	fig = plt.figure(figsize=(800/my_dpi, 1000/my_dpi), dpi=my_dpi)
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	
	nj,ni = np.shape(PSIrain)
	outc = np.zeros((nj,ni))

	for j in range(nj):
		for i in range(ni):
			outc[j,i] = encode_data_generic(PSIrain[j,i],levels)
	
	VMIN = 0 # after encoding rain value begins from 1 ("0" is white color)
	VMAX = 14 # after encoding rain value end after 14 
	outc[outc<=0.] = np.nan
	outm = ma.masked_where(np.isnan(outc),outc)
	pm = m.pcolor(Xinca,Yinca,outm,vmin=VMIN,vmax=VMAX)
	pm.set_cmap(cmRain) # change to user colorbar
	pm.cmap.set_over([80./255.,0.,80./255.])
	pm.cmap.set_under('white')

	cbar = plt.colorbar(pm,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],extend='both')  
	cbar.set_ticklabels(levels)
	cbar.set_label('mm/hr',fontsize=16)

	VMIN = 0 # after encoding rain value begins from 1 ("0" is white color)
	VMAX = 1 # after encoding rain value end after 14 

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
		
	ft = date_analysis+datetime.timedelta(minutes=(timestep+fstep*timestep))

	title = 'Stochastic ensemble member %02d forecast (out of %d)\n %s (+%d min)' % (ens_mem+1,n_ens_members,ft.strftime('%d/%m/%Y %H:%M'),(timestep+fstep*timestep))
	plt.suptitle(title,fontsize=18)

	#plt.show()
	
	fout = '%s/ens_forecast_%02d.png' % (PATHOUT,fstep+1)
	plt.savefig(fout,dpi=my_dpi)	
	plt.close()



	return 0



########
# main #
########

if __name__ == "__main__":

	start = time.time()

	# define global constants
	convert_to_mmhr = 1

	n_leadtimes = 24 
	timestep = 5

	if(convert_to_mmhr): 
		n_leadtimes = 12 
		timestep = 10	

	ENSEMBLE_ARCHIVE = '../STEPS_FORECAST'

	missval = -999.

	# get precipitation map grid coords
	lola = np.loadtxt('../Config/inca_lonlat.asc')
	Lon = lola[:,::2] # get Longitude
	Lat = lola[:,1::2] # get Latitude

	NYinca,NXinca = np.shape(Lat)

	# load color map
	H = scipy.io.loadmat('../Config/dbzbar.mat')
	cmR = H['dbzbar']
	ll = len(cmR)
	cmRain = cmR[1:ll-1,:]
	cmRain = mpl.colors.ListedColormap(cmRain)

	# precipitation levels:
	levels  = [0.05,0.1,0.2,0.7,1.2,2.,4.,6.,9.,13.,18.,24.,30.,50.,100.]
	
	leftLon = 33.5 # Lon.min()
	leftLat = 28.8 #Lat.min()
	rightLon =  36.5 #Lon.max()
	rightLat = 34. #Lat.max()
	
	date_analysis = datetime.datetime(2018,4,26,12,0)

	n_ens_members = 50

	R_o = read_ensemble_forecast(date_analysis)	

	# The output of one ensemble member is array with empty cells
	# we use np.isnan to fill the values
	
	R_o[np.isnan(R_o)] = missval

	# create basic mapping data
	# define map
	m = Basemap(llcrnrlon=leftLon,llcrnrlat=leftLat,urcrnrlon=rightLon,urcrnrlat=rightLat,projection='mill',resolution='h')
	Xinca, Yinca = m(Lon, Lat) # get x & y from the source map

	# get Borders coordinates
	# Israel borders data Base
	D = scipy.io.loadmat('../Config/borders_data_base.mat')
	borders = D['borders']
	Xborder,Yborder = m(borders[:,0],borders[:,1]) # get x & y from the politic borders data base

	for ens_mem in range(8):

		PATHOUT = '../GIF_STEPS_%dmin/GIF_ens_mem_%d' % (timestep,ens_mem+1)
		os.system('mkdir -p %s' % PATHOUT)
	
		print("Ensemble member",ens_mem+1)

		#### parrallel computing 
		pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
		pool.map(plotFigure_ens_member, range(n_leadtimes)) 
		pool.close()
		pool.join()   
		print('done')
		####	
		

	end = time.time()
	print("Program finished after: %2.1f [minutes]" % ((end-start)/60.))

