#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:40:30 2026

@author: kmc249
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
import astroalign as aa
from pathlib import Path
from smith_utils import *
from lookup_name import *

#to align and trim images
def register_with_flips(src, ref, **kwargs):
    """
    Try registering src → ref with astroalign.
    If it fails, try horizontal, vertical, and both flips.
    Returns aligned image and footprint, or (None, None) if all fail.
    """

    flip_funcs = [
        lambda x: x,                     # no flip
        np.fliplr,                       # horizontal
        np.flipud,                       # vertical
        lambda x: np.flipud(np.fliplr(x))# both
    ]

    for func in flip_funcs:
        try:
            test_img = func(src)
            aligned, footprint = aa.register(test_img, ref, **kwargs)
            return aligned, footprint
        except Exception as exc:
            print(exc)
            pass

    raise RuntimeError("astroalign failed for all flip orientations.")


overwrite=False

#loop
for target in ['GROJ1655-40']:#SWIFT_list:#['GX339-4']:#xrb_list:
    print(target)
    try:
        txtfile = glob.glob(f'/neta/xrb/{target}/temp/{target}*.txt')[0]
        data = {}
        with open(txtfile) as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                key, value = line.split(':', 1)

                data[key.strip()] = value.strip()

        most_files_band = data.get('band')
        ref = data.get('reference_image')
        x_pixel = float(data.get('x_pixel'))
        y_pixel = float(data.get('y_pixel'))
        print(x_pixel, y_pixel)
    except:
        print(f'Skipped {target}')
        continue
    #data and 512x512 cutout
    IM=fits.getdata(ref)
    
    size = 512
    hs = size // 2 
    cutout = IM[int(y_pixel)-hs:int(y_pixel)+hs, int(x_pixel)-hs:int(x_pixel)+hs]
    '''
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(IM)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,10))
    plt.imshow(IM, cmap='gray', origin='lower', norm=norm)
    plt.scatter(x_pixel, y_pixel, marker='x')
    plt.show()
    
    
    #cutout should be Roughly centered on the star
    vmin, vmax = interval.get_limits(cutout)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,10))
    plt.imshow(cutout, cmap='gray', origin='lower', norm=norm)
    plt.scatter(256, 256, marker='x')
    plt.title(target)
    plt.show()
    '''
    #Pull all the files
    basedir = Path(f'/neta/xrb/{target}')
    filelist = [
        f for f in basedir.glob('*/opt/rccd/*/*')
        if '_trimmed' not in str(f)
    ]
    
    #Make a df to keep track
    fwhms = pd.DataFrame(columns=[
    'filename','telescope','filter', 
    'maxshift', 'bad'
    ])
    
    rows = []

    for file in filelist:
        p = Path(file)
        
        filter_ = p.parent.name
        telescope = p.parents[3].name
        trim_path = (p.parent.parent / f"{p.parent.name}_trimmed" / f"trim_{p.name}")
        rows.append({
            'filename': str(p),
            'trim_filename': str(trim_path),
            'telescope': telescope,
            'filter': filter_,
        })
    
    fwhms = pd.concat([fwhms, pd.DataFrame(rows)], ignore_index=True)
    
    
    #For each file: 
    for id, row in fwhms.iterrows():
        print(f'working on {id} out of {len(fwhms)}')
        outfile = Path(row['trim_filename'])
        file=row['filename']
        print(file)
        
        #check if a trimmed version already exists, and skip if we want to
        if not overwrite and outfile.exists():
            print('trim file exists!')
            fwhms.at[id, 'bad']=False
            continue
        
        #make sure the trim dir exists
        outfile.parent.mkdir(exist_ok=True)
        
        #pull the file data
        try:
            im,hdr = fits.getdata(file,header=True)
        except:
            print('could not read ', file)
            fwhms.at[id, 'bad']=True
            continue
            
        #align and trim, save trimmed images
        img = np.asarray(IM, dtype='<f8')
        inp_img = np.asarray(fits.getdata(file), dtype='<f8')
        '''
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(inp_img)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
        plt.figure(figsize=(10,10))
        plt.imshow(inp_img, cmap='gray', origin='lower', norm=norm)
        plt.show()
        '''
        #manually flipping it
        try:
            img_aligned, footprint = register_with_flips(inp_img, img, detection_sigma=5.0, max_control_points=75)
            # now, apply the saved transform to master
            trimimg=img_aligned[int(y_pixel)-hs:int(y_pixel)+hs, int(x_pixel)-hs:int(x_pixel)+hs]
            master=img
        except:
            print('bad alignment :(')
            print('trying smith way first')
            try:
                #try shifting it first and see what happens
                xshift, yshift = cross_image(img, inp_img, int(np.shape(IM)[0]/2), int(np.shape(IM)[1]/2), boxsize=400)
                newimg=shift_image(inp_img,xshift, yshift)
                #set some buffer around just in case
                buff=40
                second_inp=newimg[int(y_pixel)-hs-buff:int(y_pixel)+hs+buff, int(x_pixel)-hs-buff:int(x_pixel)+hs+buff]
                '''
                plt.figure(figsize=(10,10))
                plt.imshow(second_inp, cmap='gray', origin='lower', norm=norm)
                plt.show()
                '''
                img_aligned, footprint = register_with_flips(second_inp, np.asarray(cutout, dtype='<f8'), detection_sigma=2.0, max_control_points=75)
                trimimg=img_aligned
                master=cutout
                print('smith way worked!')
            except:
                print('no hope.')
                fwhms.at[id, 'bad']=True
                continue
            

        
        
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(inp_img)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
        interval1 = ZScaleInterval()
        vmin2, vmax2 = interval1.get_limits(master)
        norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
        '''
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(inp_img, cmap='gray', interpolation='none', origin='lower', norm=norm)
        axes[0, 0].axis('off')
        axes[0, 0].set_title("Source Image")
        
        axes[0, 1].imshow(master, cmap='gray', interpolation='none', origin='lower', norm=norm2)
        axes[0, 1].axis('off')
        axes[0, 1].set_title("Master Image")

        axes[1, 0].imshow(img_aligned, cmap='gray', interpolation='none', origin='lower', norm=norm)
        axes[1, 0].axis('off')
        axes[1, 0].set_title("Result of aligning w/Example")

        axes[1, 1].imshow(trimimg, cmap='gray', interpolation='none', origin='lower', norm=norm)
        axes[1, 1].axis('off')
        axes[1, 1].set_title("Final Trim")
        
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.show()
        '''
        #save to trim:
        hdr['TRIM']=True
        filename=file.split('/')[-1]
        fits.writeto(outfile, trimimg, hdr, overwrite=True, output_verify='ignore')
        fwhms.at[id, 'bad']=False

    #save the fwhms df for future reference 
    fwhms.to_csv(f'/neta/xrb/{target}/temp/{target}_trim_log.csv', index=False)
    print((fwhms['bad'] == False).sum())
