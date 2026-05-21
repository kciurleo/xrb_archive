#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:10:53 2026

@author: kmc249
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#optical is saved in nm, IR in microns
for filt in ['V','R','I']:
    df = pd.read_csv(
        f"/home/kmc249/Downloads/KPNO_{filt}.txt",
        comment="#",
        delim_whitespace=True,
        names=["lambda_nm", "trans_percent"]
    )

        
    #to microns
    df["lambda_um"] = df["lambda_nm"] / 1000.0
    
    #to fraction
    df["T"] = df["trans_percent"] / 100.0
    
    lam = df["lambda_um"].values
    T = df["T"].values
    
    #central wavelength
    lambda_c = np.trapezoid(lam * T, lam) / np.trapezoid(T, lam)
    
    #print(f"{filt} central wavelength = {lambda_c:.4f} microns")
    print(f"{lambda_c:.4f},")
    plt.plot(lam, T)
    plt.scatter(lam, T)
    
for filt in ['J','H','K']:
    df = pd.read_csv(
        f"/home/kmc249/Downloads/{filt}_andi.txt",
        comment="#",
        delim_whitespace=True,
        names=["lambda_um", "trans_percent"]
    )
    
    #to fraction
    df["T"] = df["trans_percent"] / 100.0
    
    lam = df["lambda_um"].values
    T = df["T"].values
    
    #central wavelength
    lambda_c = np.trapezoid(lam * T, lam) / np.trapezoid(T, lam)
    
    #print(f"{filt} central wavelength = {lambda_c:.4f} microns")
    plt.plot(lam, T)
    plt.scatter(lam, T)
    print(f"{lambda_c:.4f},")