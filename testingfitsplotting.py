

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 10:37:31 2025

First version: Wed Feb 7 15:55:54 2024

Script to represent one or multiple FITS images in a single plot. Images must be 
one-dimensional in frequency, i.e. continuum, collapsed moment, single channel.

Images can be raster or contours, as in CASA's imview or CARTA. Regions in 
crtf format are also supported.

Used for Figures 1, 2, 4 and A.1 in Martínez-Henares et al. 2025, A&A, 669A 
(FAUST XXV).

@author: Antonio Martínez-Henares https://github.com/amarhen 
"""

##############################################
### Imports
##############################################
from astropy.io import fits
from astropy import wcs
from regions import PixCoord, RectanglePixelRegion, SkyRegion, Regions, RectangleSkyRegion

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib
import matplotlib.colors as colors

##############################################
### Classes and functions
##############################################

class FITSmap():
    """
    This class represents a map from a FITS file with no frequency axis, i.e.
    a continuum, moment zero, etc... map. 
    
    Attributes
    ------------
    file : str
        The path to the FITS file.
    flux_units : str
        The flux units of the map, typically Jy beam^{-1], or Jy beam^{-1] km s^{-1}
        if it is a moment zero map.
    raref : float
        Right ascension of the source, used to represent in relative coordinates.
    decref : float
        Declination of the source, used to represent in relative coordinates.
    rms : float
        RMS noise of the map, used to plot contours.
        Should be measured previously with CASA, CARTA, etc.
    flux : numpy array
        Two-dimensional array containing the flux density values of the map.
    ra : numpy array
        Two-dimensional array containing the right ascension values of the map.
    dec : numpy array
        Two-dimensional array containing the declination values of the map.
    rarel : numpy array
        Two-dimensional array containing the relative right ascension values of the map.
    decrel : numpy array
        Two-dimensional array containing the relative declination values of the map.
    bmaj : float
        Major axis of the synthesized beam.
    bmin : float
        Minor axis of the synthesized beam.
    bpa : float
        Position angle of the synthesized beam.
    w : WCS instance 
        WCS of the FITS file, used for plotting pixel regions.
    
        
    Methods
    ------------
    read_fits():
        Reads parameters from the FITS file and stores them as attributes.
    
    """
    
    
    def __init__(self, file, flux_units, raref = 0.0, decref = 0.0, rms = 0.0):
        
        self.rms = rms
        self.flux_units = flux_units
        self.raref = raref
        self.decref = decref
        
        flux, ra, dec, rarel, decrel, bmaj, bmin, bpa, w, header = self.read_fits(file)
        
        self.flux = flux
        self.ra = ra
        self.dec = dec
        self.rarel = rarel
        self.decrel = decrel
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.w = w
        self.header = header

    def read_fits(self, file):
        """
        Method to read and compute parameters of the FITS file.

        Parameters
        ----------
        file : str
            The path to the FITS file.

        Returns
        -------
        fluxi : numpy array
            Two-dimensional array containing the flux density values of the map.
        rai : numpy array
            Two-dimensional array containing the right ascension values of the map.
        deci : numpy array
            Two-dimensional array containing the declination values of the map.
        rareli : numpy array
            Two-dimensional array containing the relative right ascension values of the map.
        decreli : numpy array
            Two-dimensional array containing the relative declination values of the map.
        bmaj : float
            Major axis of the synthesized beam.
        bmin : float
            Minor axis of the synthesized beam.
        bpa : float
            Position angle of the synthesized beam.
        w : WCS instance 
            WCS of the FITS file, used for plotting pixel regions.
        header : FITS header
            Header of the FITS file in the astropy.io fits format.

        """
        
        # Get header
        
        hdulist = fits.open(file)
        header = hdulist[0].header
        
        # Get flux
        
        if len(np.shape(hdulist[0].data)) == 4:
            fluxi = hdulist[0].data[0][0]
        elif len(np.shape(hdulist[0].data)) == 3:
            fluxi = hdulist[0].data[0]
        elif len(np.shape(hdulist[0].data)) == 2:
            fluxi = hdulist[0].data
            
        # Build coordinate axis in pixel units
        
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        crval1 = header['CRVAL1'] # RA of reference pixel
        crval2 = header['CRVAL2'] # DEC of reference pixel
        cdelt1 = header['CDELT1']  # Pixel increment in RA, DEG
        cdelt2 = header['CDELT2']  # Pixel increment in DEC, DEG
        try:
            bmaj = header['BMAJ']
            bmin = header['BMIN']
            bpa = header['BPA']
        except(KeyError):
            bmaj = 0.0
            bmin = 0.0
            bpa = 0.0
        
        ra_axis = np.linspace(start = crval1 - cdelt1*(naxis1/2),
                            stop = crval1 + cdelt1*(naxis1/2),
                            num=naxis1
                            )
        
        dec_axis = np.linspace(start = crval2 - cdelt2*(naxis2/2),
                            stop = crval2 + cdelt2*(naxis2/2),
                            num=naxis2)
        
        rai, deci = np.meshgrid(ra_axis, dec_axis)
        
        rarel_axis = (ra_axis - self.raref)*3600 #arcsec
        decrel_axis = (dec_axis - self.decref)*3600 #arcsec
        rareli, decreli = np.meshgrid(rarel_axis, decrel_axis)
        
        bmaj = bmaj*3600
        bmin = bmin*3600
        
        w = wcs.WCS(hdulist[0].header, naxis=2)
        
        return fluxi, rai, deci, rareli, decreli, bmaj, bmin, bpa, w, header
    
class MapForPlot():
    """
    This class represents the plot itself, with the information from the FITS
    file plus custom parameters for the plot.
    
    Attributes
    ------------
    fitsmap : FITSmap instance
        Contains all relevant parameters form the FITS file: flux density,
        coordinates, beam, ...
    imtype : str
        Raster or contours.
    colorscale : str
        Colorscale in case of raster image. Either a matplotlib colorscale or
        a custom-defined one.
    custom_cmap : str
        If customized colormap (i.e. reduced range of colours of a certain 
        colormap for better visualization/aesthetics), indicates the Matplotlib
        colormap that will be used.
    custom_cmap_min : float
        Mininum value of the custom colormap, between 0 and 1. 
    custom_cmap_max : float
        Maximum value fo the custom colormap, between 0 and 1.
    colorbar : bool
        If True, plots a colorbar on the side.
    cont_levels : list
        List of contour levels for contour maps.
    cont_thickness : float
        Contour thickness.
    cont_color : str
        Contour color.
    plot_beam : bool
        If True, plots the beam ellipse on the lower left.
    beam_color : str
        Color of the beam ellipse. Default is black.
    beam_background : bool
        If True, a square shadow is plotted behind the beam ellipse for better contrast.
    plot_scalebar : bool
        If True, plots the scalebar.
    scalebar_contrast : bool
        If True, a square shadow is plotted behind the scalebar for better contrast.
    
    """
    
    def __init__(self, fitsmap, imtype, colorscale = None, custom_cmap = None,
                 custom_cmap_min = 0.0, custom_cmap_max = 1.0, colorbar = None, 
                 cont_levels = None, cont_thickness = None, cont_color = None,  
                 plot_beam = False, beam_color = 'black', beam_background = False, 
                 plot_scalebar = False, scalebar_contrast = False):
        
        self.fitsmap = fitsmap
        self.imtype = imtype
        self.colorscale = colorscale
        self.custom_cmap = custom_cmap
        self.custom_cmap_min = custom_cmap_min
        self.custom_cmap_max = custom_cmap_max
        self.colorbar = colorbar
        self.cont_levels = cont_levels
        self.cont_thickness = cont_thickness
        self.cont_color = cont_color
        self.plot_beam = plot_beam
        self.beam_color = beam_color
        self.beam_background = beam_background
        self.plot_scalebar = plot_scalebar
        self.scalebar_contrast = scalebar_contrast

def plot_maps(mapforplot_list, plotparams, region = None, save = False, 
              savepath = None, savename = 'figure.png', mask_vlim = False):
    """
    Function to plot the FITS maps into a single image with custom options.

    Parameters
    ----------
    mapforplot_list : list of MapForPlot instances
        List that includes the different FITS maps to be plotted.
    plotparams : dict
        Dictionary containing plotting parameters.
    region : Region instance, optional
        CASA/CARTA Region to be plotted, e.g. square, polygon... 
    save : bool, optional
        If True, saves the plot in png format.
    savepath : str
        Path to save the image. Default is current folder.
    savename : str
        Name of the figure. Default is figure.png.
    mask_vlim : bool, optional
        If true, applies a mask on the flux density.

    Returns
    -------
    None.

    """
    
    # Initiate figure with custom parameters
    plt.figure(figsize=[12,10])
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    
    xaxis_font = {'size':plotparams['fontsize'], 'weight':'normal'}
    yaxis_font = {'size':plotparams['fontsize'], 'weight':'normal'}
    plt.tick_params(axis='both', which='major', labelsize=plotparams['fontsize'], 
                    length=12, width=1 
                    )
    
    plt.xlabel('Relative RA (arcsec)', **xaxis_font)
    plt.ylabel('Relative DEC (arcsec)', **yaxis_font)
    
    plt.ylim([plotparams['ymin'],plotparams['ymax']])
    plt.xlim([plotparams['xmax'],plotparams['xmin']])
    
    if (plotparams['ymax'] > 10 and plotparams['xmin'] < -10):
        plt.yticks([-10,-5,0,5,10])
        plt.xticks([-10,-5,0,5,10])     
    elif(plotparams['ymax'] > 4 and plotparams['xmin'] > -4 and plotparams['xmax'] < 4):
        plt.yticks([-4,-2,0,2,4])
        plt.xticks([-4,-2,0,2,4])
    elif(plotparams['ymax'] > 2 and plotparams['xmin'] > -2 and plotparams['xmax'] < 2):
        plt.yticks([-2,0,2])
        plt.xticks([-2,0,2])
    elif(plotparams['xmin'] >= -5 and plotparams['xmax'] <= 15 and plotparams['ymin'] < -5):
        plt.yticks([-10, -5, 0, 5, 10])
        plt.xticks([-5, 0, 5, 10, 15])
    elif(plotparams['xmin'] > -3.1 and plotparams['xmax']<3.1):
        plt.yticks([-2, -1, 0, 1, 2])
        plt.xticks([-2, -1, 0, 1, 2])
    
    # Plot the list of maps
    for mapforplot in mapforplot_list:
        
        # Raster maps
        if mapforplot.imtype == 'raster':
            if mapforplot.colorscale == 'custom':
                colormap = truncate_colormap(plt.get_cmap(mapforplot.custom_cmap), 
                                             mapforplot.custom_cmap_min, mapforplot.custom_cmap_max)
                mapforplot.colorscale = colormap
            try:
                if plotparams['scale'] == 'log':
                        sc = plt.scatter(mapforplot.fitsmap.rarel, mapforplot.fitsmap.decrel, 
                                 c=mapforplot.fitsmap.flux, cmap=mapforplot.colorscale,
                                 norm=matplotlib.colors.LogNorm(vmin=plotparams['vmin'], vmax=plotparams['vmax']))
                elif plotparams['scale'] == 'linear':
                    if mask_vlim:
                        masked_flux = np.copy(mapforplot.fitsmap.flux)
                        for i in range(len(masked_flux)):
                            for j in range(len(masked_flux[i])):
                                if (masked_flux[i][j] <= plotparams['maskmin'] or masked_flux[i][j] >= plotparams['maskmax']):
                                    masked_flux[i][j] = np.nan
                        try:
                            sc = plt.scatter(mapforplot.fitsmap.rarel, mapforplot.fitsmap.decrel, 
                                     c=masked_flux, cmap=mapforplot.colorscale,
                                     vmin = plotparams['vmin'], vmax=plotparams['vmax'])
                        except(KeyError):
                            sc = plt.scatter(mapforplot.fitsmap.rarel, mapforplot.fitsmap.decrel, 
                                     c=masked_flux, cmap=mapforplot.colorscale)
                    else:
                        sc = plt.scatter(mapforplot.fitsmap.rarel, mapforplot.fitsmap.decrel, 
                                 c=mapforplot.fitsmap.flux, cmap=mapforplot.colorscale,
                                 vmin = plotparams['vmin'], vmax=plotparams['vmax'])
            except(KeyError): # in case I forget
                sc = plt.scatter(mapforplot.fitsmap.rarel, mapforplot.fitsmap.decrel, 
                                 c=mapforplot.fitsmap.flux, cmap=mapforplot.colorscale)
            
            if mapforplot.colorbar:
                cb = plt.colorbar(sc)
                # workaround to be able to modify the fontsize of the tick labels in the colorbar
                cb.set_label(label=mapforplot.fitsmap.flux_units,rotation=270, labelpad=plotparams['fontsize']*1.1, size=plotparams['fontsize'])
                ticks = cb.get_ticks()
        
        # Contour maps
        elif mapforplot.imtype == 'contours':
            
            levels = [i * mapforplot.fitsmap.rms for i in mapforplot.cont_levels]
            plt.contour(mapforplot.fitsmap.rarel, mapforplot.fitsmap.decrel, 
                        mapforplot.fitsmap.flux, levels = levels, 
                        colors = mapforplot.cont_color, linewidths = mapforplot.cont_thickness,
                        negative_linestyles='dashed')
        
        # Beam ellipse
        if mapforplot.plot_beam:
            if mapforplot.beam_color == 'blue':
                beam_ellipse = Ellipse((plotparams['xmax']*0.9, plotparams['ymin']*0.9), mapforplot.fitsmap.bmin, mapforplot.fitsmap.bmaj,
                              -mapforplot.fitsmap.bpa, color='blue', fill = True, zorder=2)
            elif mapforplot.beam_color == 'red':
                beam_ellipse = Ellipse((plotparams['xmax']*0.9, plotparams['ymin']*0.9), mapforplot.fitsmap.bmin, mapforplot.fitsmap.bmaj,
                              -mapforplot.fitsmap.bpa, color='red', fill = True, zorder=2)
            elif mapforplot.beam_color == 'black':
                beam_ellipse = Ellipse((plotparams['xmax']*0.9, plotparams['ymin']*0.9), mapforplot.fitsmap.bmin, mapforplot.fitsmap.bmaj,
                              -mapforplot.fitsmap.bpa, color='black', fill = True, zorder=2)
            elif mapforplot.beam_color == 'white':
                beam_ellipse = Ellipse((plotparams['xmax']*0.9, plotparams['ymin']*0.9), mapforplot.fitsmap.bmin, mapforplot.fitsmap.bmaj,
                              -mapforplot.fitsmap.bpa, color='white', fill = True, zorder=2)
            elif mapforplot.beam_color == 'darkred':
                beam_ellipse = Ellipse((plotparams['xmax']*0.9, plotparams['ymin']*0.9), mapforplot.fitsmap.bmin, mapforplot.fitsmap.bmaj,
                              -mapforplot.fitsmap.bpa, color='darkred', fill = True, zorder=2)
            elif mapforplot.beam_color == 'cyan':
                beam_ellipse = Ellipse((plotparams['xmax']*0.9, plotparams['ymin']*0.9), mapforplot.fitsmap.bmin, mapforplot.fitsmap.bmaj,
                              -mapforplot.fitsmap.bpa, color='cyan', fill = True, zorder=2)
            # below is to have a shadowed square behind the beam ellipse for better contrast, see Fig. 1 bottom panels in Martínez-Henares et al. (2025)
            if mapforplot.beam_background:
                someX, someY = plotparams['xmax']*0.85,plotparams['ymin']*0.95
                currentAxis = plt.gca()
                currentAxis.add_patch(Rectangle((someX, someY), 1,1.,
                                  alpha=0.4,facecolor='black'))
            plt.gca().add_patch(beam_ellipse)
    
    # Plot CASA regions or rectangles, e.g. for insets
    if region is not None:
        if isinstance(region, RectanglePixelRegion):
            region.plot(facecolor='none', edgecolor = plotparams['region_color'], lw = 1, alpha=0.5)
        else:
            region_relative = sky_to_relative(region, mapforplot_list[1].fitsmap)
            region_relative.plot(color=plotparams['region_color'], lw=2.0)
            

    # Plot scalebar. Since it is distance-dependent, user has to specify limits in plotparams.
    if mapforplot.plot_scalebar:
        # text
        tx = plt.text(-(abs(abs(plotparams['scalebar_left'])-abs(plotparams['scalebar_right'])))/2 + plotparams['scalebar_left'],
                      plotparams['ymin']*0.85, 
                      plotparams['scalebar_au_size'],size=18, color='white',
                      horizontalalignment='center',
                      verticalalignment='center'
                      )
        # bar
        tx = plt.hlines(y=plotparams['ymin']*0.9, xmin=plotparams['scalebar_left'], 
                        xmax=plotparams['scalebar_right'], linewidth=2, 
                        color=plotparams['scalebar_color'])
        
        if mapforplot.scalebar_contrast:
            someX, someY = -plotparams['xmax']*0.9,plotparams['ymin']*0.95
            currentAxis = plt.gca()
            currentAxis.add_patch(Rectangle((someX, someY), 
                                            abs(abs(plotparams['scalebar_right']
                                                    -plotparams['scalebar_left']))*1.2,
                                            2.,
                                            alpha=0.4,facecolor='black'))
            
    # Additionally, the user might be interested in plotting a textbox. 
    # Since this is highly dependent on the specific case, it is not included as an option. 
    # Nevertheless, below an example of how a textbox would be implemented.
    
    # tx = plt.text(10.,7.6,r'H$^{13}$CO$^+$ (3.78 to 4.62 km s$^{-1}$)     ' + '\n' + '\n',color='red', size=22, 
    #             bbox=dict(facecolor='white', edgecolor='black', pad=6.0))
    # tx = plt.text(10.,8.3,r'H$^{13}$CO$^+$ (2.07 to 2.92 km s$^{-1}$)',color='blue', size=22)
    # tx = plt.text(10.,7.3,r'SiO (−0.48 to 7.57 km s$^{-1}$)',color='black', size=22)
    
    if save:
        if savepath is not None:
            plt.savefig(savepath + '/' + savename, bbox_inches='tight', dpi=50)
        else:
            plt.savefig(savename, bbox_inches='tight', dpi=50)
        plt.close()
    else:
        plt.show()
    
def sky_to_relative(region, fits_reference):
    """
    Convert a Region from sky/pixel coordinates to pixel coordinates in the 
    relative frame of coordinates.

    Parameters
    ----------
    region : Region instance
        CASA/CARTA Region to be plotted, e.g. square, polygon... 
    fits_reference : FITSMap instance
        FITSMap instance to have the reference for header and WCS.

    Returns
    -------
    pixel_region : Region instance
        In relative pixel coordinates, for plotting in plot_maps().

    """
    
    header = fits_reference.header
    
    if isinstance(region, RectanglePixelRegion):
        region.center.x = (region.center.x-header['CRPIX1']) * header['CDELT1'] * 3600
        region.center.y = (region.center.y-header['CRPIX2']) * header['CDELT2'] * 3600
        region.width = region.width * np.sqrt(header['CDELT1']**2.0) * 3600 / 2
        region.height = region.height * np.sqrt(header['CDELT2']**2.0) * 3600 / 2
        pixel_region = region
    else:
        pixel_region = region.to_pixel(fits_reference.w)
        for i in range(len(pixel_region.vertices.x)):
            pixel_region.vertices.x[i] = (pixel_region.vertices.x[i]-header['CRPIX1']) * header['CDELT1'] * 3600
            pixel_region.vertices.y[i] = (pixel_region.vertices.y[i]-header['CRPIX2']) * header['CDELT2'] * 3600

    return pixel_region

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Function to create a custom colormap by truncating a matplotlib colormap.
    
    From https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib 

    Parameters
    ----------
    cmap : str
        Matplotlib colormap.
    minval : float, optional
        Minimum of the original colormap to go into the new one. The default is 0.0.
    maxval : float, optional
        Maximum of the original colormap to go into the new one. The default is 1.0.
    n : int, optional
        Number of steps in the colormap. The default is 100.

    Returns
    -------
    new_cmap : matplotlib colormap
        Custom colormap.

    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#%%

# The user should edit options below here.

##############################################
### Global parameters
##############################################

# Coordinates of the (0,0) arcsec position - in this example, [BHB2007]11
raref = 257.8462667 # deg
decref = -27.40914167 # deg

plotparams = {'xmin':-11, # Limits of the plot, in relative arcsec
              'xmax':11,
              'ymin':-11,
              'ymax':11,
              'fontsize':20, # Fontsize of labels
              'vmin':-0.01, # To tune the contrast of the colorbar
              'vmax':0.05,
              'scale':'linear', # Linear or logarithmic ('log') colorscale
              'region_color':'black', # In case of plotting a Region
              'maskmin':0.005, # In case of masking values 
              'maskmax':1e6, # i.e. don't mask the max values.
              'scalebar_left': -6.5, # Left end of the scalebar. Depends on distance, so user should compute it.
              'scalebar_right':-9.625,
              'scalebar_au_size':'500 au',
              'scalebar_color': 'white'}


##############################################
### Main
##############################################
import pandas as pd
zoomphot=pd.read_csv('/home/kmc249/current_best_J_grid_fit.csv')
opticalinits=pd.read_csv('/home/kmc249/current_best_R_grid_fit.csv')

#IR image
bkg_sub_full_data,hdr = fits.getdata('/neta/xrb/AqlX-1/temp/Aql_X-1_J_stack_aligned_to_opt.fits',header=True)
res=2
zoomdata=bkg_sub_full_data[2*231:2*281,2*231:2*281]

#opt image
imdataopt,hdropt = fits.getdata('/home/kmc249/Downloads/hires_master.fits',header=True)
imdatazoom=imdataopt[231*res:281*res,231*res:281*res]


def main():
    
    # EXAMPLE SESSION:
    
    ### Define the maps and their parameters.
    
    # First search for the FITS file, and state the units and rms noise.
    cont_fits = FITSmap(file='/neta/xrb/AqlX-1/temp/Aql_X-1_J_stack_aligned_to_opt.fits', 
                        flux_units = r'Jy beam$^{-1}$', 
                        raref = raref, decref = decref, rms=0.14e-3)
    
    # Then create the plot of the FITS file. It can be raster or contours.
    cont_plot = MapForPlot(fitsmap = cont_fits, 
                            imtype = 'contours', 
                            cont_levels = [20,30,40,50,100,200], 
                            cont_thickness = 1.5, 
                            cont_color = 'white',
                            plot_beam = False)
    
    # Define raster + contours for an SiO moment-zero image.
    sio_mom0_fits = FITSmap(file='/home/kmc249/Downloads/hires_master.fits',
                            flux_units = r'Jy beam$^{-1}$ km s$^{-1}$', 
                            raref = raref, decref = decref, rms=5.0e-3)
    
    sio_mom0_raster_plot = MapForPlot(fitsmap = sio_mom0_fits,
                                      imtype = 'raster',
                                      colorscale = 'custom',
                                      custom_cmap = 'turbo', # In case of custom colormap, see matplotlib colormaps
                                      custom_cmap_min = 0.15,
                                      custom_cmap_max = 1.0,
                                      colorbar = True,
                                      plot_beam = True,
                                      beam_color = 'black',
                                      beam_background = True,
                                      plot_scalebar = True,
                                      scalebar_contrast = True) # Note that only makes sense to define scalebar in one of the maps. 
                                                                # It must be the last element of the matforplot_list later on to not be overlaid by another image.

    
    # Here in case I wanted to plot a polygon region from CASA (e.g. Fig. 1 in 
    # Martínez-Henares et al. 2025):
    # red_poly_regions = Regions.read('region.crtf', format='crtf')
    # red_poly_region = red_poly_regions[0]
    
    # Rectangle region for inset
    square_region = RectanglePixelRegion(PixCoord(x=0, y=0), width=9,
                                height=9)
    
    # Creat the list of images and append the plots. Last one is the one with
    # the scalebar information
    mapforplots_list = []
    mapforplots_list.append(cont_plot)
    mapforplots_list.append(sio_mom0_raster_plot) # The last one includes scalebar parameters
    
    # Now call the plot_maps function with the list of images and regions
    plot_maps(mapforplots_list, 
              plotparams, 
              region = square_region, # If it were the polygon region, red_poly_region
              save = False, 
              savepath = '/home/kmc249/Downloads', # Don't put / at the end
              savename = 'test.png', 
              mask_vlim = True)
    
if __name__ == "__main__": # If not executed via import in other python script.
    main()