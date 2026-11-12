#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:41:03 2026

@author: kmc249
"""


from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch

BASE_DIR = Path("/neta/xrb")
OUTPUT_DIR = Path("/home/kmc249/test_data/xrb_archive/")

PLOTS_DIR = OUTPUT_DIR / "websiteplots"
bands=['B','V','R','I']
EXCLUDED_DIRS = {
    "IRCALIBS",
    "META",
    "PRODUCTS",
    "MASTERCALIBS",
    "MASTERCALIBS_UNBINNED", "REPLOGS",
    '.Trash-1566', '.Trash-1571'
}


def get_targets():
    return sorted([
        p.name for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_DIRS
    ])


def stack_fits_plot(fits_file, output_file, target, band):

    if not fits_file.exists():
        return False

    try:
        #save and plot
        header=fits.getheader(fits_file)
        stacked=fits.getdata(fits_file)
        header['STACKED']='True'
        numimgs=header['NUM_IMS']
        
        

        
        # ============================================================
        # PLOT STACK
        # ============================================================
        
        interval=ZScaleInterval()
        
        vmin,vmax=interval.get_limits(
            stacked
        )
        
        
        norm=ImageNormalize(
            vmin=vmin,
            vmax=vmax,
            stretch=AsinhStretch()
        )
        
        
        plt.figure(figsize=(10,10))
        
        plt.imshow(
            stacked,
            cmap='gray_r',
            origin='lower',
            norm=norm
        )
        
        plt.title(
            f'{target} {band} best seeing stack\n'
            f'{numimgs} images'
        )
        
        plt.axis('off')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        

        return True

    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = get_targets()

    manifest = {
        "targets": []
    }

    for target in targets:

        print(f"Processing {target}...")

        target_info = {
            "name": target,
            "products": {}
        }
        #STACKED IMAGES
        for band in bands:
            r_fits = (
                BASE_DIR
                / "PRODUCTS"
                / "stacked_images_optical"
                / f"{target}_stacked_{band}.fits"
            )
    
            r_png = (
                PLOTS_DIR
                / target
                / f"stacked_{band}.png"
            )
    
            available = stack_fits_plot(
                r_fits,
                r_png,
                target,
                band
            )
    
            target_info["products"][f"stacked_{band}"] = {
                "available": available,
                "path": f"websiteplots/{target}/stacked_{band}.png"
                if available else None
            }
    
        manifest["targets"].append(target_info)

    # Write JSON manifest
    with open(OUTPUT_DIR / "website_products.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nFinished! Processed {len(targets)} targets.")


if __name__ == "__main__":
    main()
