#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:39:49 2026

@author: kmc249
"""
from astropy.table import Table
import pandas as pd
import numpy as np

###Drag and drop neighbor subtraction for Pagny
#Assuming your background subtracted data is called bkg_sub_full_data and your psf model is called psf_model

#This is the "relative flux factor" which you get from the stacked image; should be flux of neighbor/(ave flux of comps)
stacked_flux_factor_b=

#Calculate average flux of comp stars for this image, from running PSF photometry on the comp stars
ave_ens_flux= 

#get scaled flux of neighbor for this particular exposure
scaled_b_flux=stacked_flux_factor_b*ave_ens_flux


#using stacked x and y positions, subtract a psf of this neighbor star
#x and y positions of the neighbor:
neighbor_x=
neighbor_y=
neighbor_params=pd.DataFrame()
neighbor_params['x_0'], neighbor_params['y_0'], neighbor_params['flux']=np.array([neighbor_x]), np.array([neighbor_y]),np.array([scaled_b_flux])
neighbor_params=Table.from_pandas(neighbor_params)
test=make_model_image(np.shape(bkg_sub_full_data), psf_model, neighbor_params)

#bkg subtracted img minus the neighbor
final_data = bkg_sub_full_data - test

#Then you can do the aperture photometry on final_data


###The other way to do this would be just pulling neighbor_params from your PSF fit directly (i.e. do the fit, then pull the x, y, and flux values)
#Then you could do:
    
'''
neighbor_params=#the params you fit in this image
test=make_model_image(np.shape(bkg_sub_full_data), psf_model, neighbor_params)
final_data = bkg_sub_full_data - test
'''

#and do the aperture photometry on this.
