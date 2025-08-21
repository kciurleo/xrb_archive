#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:46:02 2025

@author: kmc249
"""

### true 2 vis plot making
path='/home/kmc249/test_data/xrb_archive/true2_tempplots/'


import io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

true2s=pd.read_csv(io.StringIO('''
CXO Name,RA,Dec,Z
2CXO J081458.3+365325,123.7429515,36.89038478,0.109137975
2CXO J081145.3+232825,122.9388137,23.47387626,0.01573402
2CXO J143232.4+340624,218.1354104,34.10690211,0.042354107
2CXO J140345.0+292143,210.9376327,29.3621805,0.06371773
2CXO J020925.1+002356,32.35479576,0.3991505407,0.06160683
2CXO J105310.9+544156,163.2955724,54.69913392,0.13650845
2CXO J124104.7+441922,190.269955,44.32295605,0.1472288
2CXO J121802.9+292440,184.5121002,29.41116256,0.10079681
2CXO J142854.2+323723,217.2258779,32.6230924,0.13048969
2CXO J235720.1-005829,359.3339162,-0.9749080614,0.33907765
2CXO J143525.3+330520,218.855417,33.08909092,0.19926393
2CXO J230817.2-094622,347.072013,-9.773037937,0.099028364
2CXO J150117.0+014958,225.3211939,1.832858213,0.128888
2CXO J150117.0+014958,225.3211939,1.832858213,0.128888
2CXO J101756.4-011430,154.4851554,-1.241770505,0.16187672
2CXO J081910.7+212609,124.7946935,21.4358111,0.0161257
2CXO J142253.3+533025,135.2719291,29.02972126,0.1940454
2CXO J090105.2+290146,132.2726832,11.24756148,0.07765917
2CXO J084905.4+111451,238.0322326,20.16452968,0.29670456
2CXO J155207.7+200952,215.722345,53.50719171,0.23531225
'''), header=0)


# Get altitude, given an ra, dec, and local sidereal time
def get_alt(ra, dec, time, lat = 19.8):
  '''
  For a given ra in decimal hours, dec in decimal degrees, LST in decimal hours,
  and lat in degrees (with Mauna Kea default), return the altitude of the object.
  '''

  HA = (time - ra)/24*360 #in degrees

  # Spherical trig calculation
  alt = 90-np.degrees(np.arccos(np.cos(np.radians(90-lat))*np.cos(np.radians(90-dec))+np.sin(np.radians(90-lat))*np.sin(np.radians(90-dec))*np.cos(np.radians(HA))))

  return(alt)

def get_airmass(ra, dec, time, lat = 41.5):
  '''
  For a given ra in decimal hours, dec in decimal degrees, LST in decimal hours,
  and lat in degrees (with Middletown default), return the airmass of the object
  and the time in decimal hours that it spends at better than 2 airmass (with a
  value of NaN if it does not rise above 2 airmass).
  '''
  z=90-get_alt(ra, dec, time, lat)
  # Get rid of all the places where the star is below the horizon
  z_visible = np.where(z<90, z, np.nan)

  airmass=1/np.cos(np.radians(z_visible))

  # Find the times only when the airmass is better than 2
  above_2 = np.where(airmass<2, time, np.nan)

  # Find how long the star is at better than 2 airmass
  time_above_2=np.round(np.nanmax(above_2)-np.nanmin(above_2),2)

  return(airmass, time_above_2)


# LST for plot
LST = np.linspace(0,24,200)


for id, row in true2s.iterrows():
    fig, axes = plt.subplots(1,2, figsize=(10,6))
    axes[0].plot(LST, get_alt(row['RA'], row['Dec'], LST, lat=33.358), label=row['CXO Name'])

    # Limits
    axes[0].set_ylim(0,90)
    axes[0].set_xlim(0,24)
    axes[0].axvline(11, color='black', linestyle='--', label='Feb/July midnight')
    axes[0].axvline(19, color='black', linestyle='--')
    axes[0].axvspan(1, 11-6, color='black', alpha=1, zorder=2)
    axes[0].axvspan(1, 11-6, color='black', alpha=1, zorder=2)
    # Text
    axes[0].legend()
    axes[0].set_xlabel("LST (hours)")
    axes[0].set_ylabel("Alt (deg)")
    
    axes[1].plot(LST, get_airmass(row['RA'], row['Dec'], LST)[0], label=row['CXO Name'])

    # Limits
    axes[1].set_ylim(1,2.5)
    axes[1].set_xlim(0,24)
    
    # Text
    
    axes[1].set_xlabel("LST (hours)")
    axes[1].set_ylabel("Airmass")
    axes[1].yaxis.set_inverted(True)
    
    axes[1].axvline(11, color='black', linestyle='--', label='Feb/July midnight')
    axes[1].axvline(19, color='black', linestyle='--')
    axes[1].axvspan(1, 11-6, color='black', alpha=1, zorder=2)
    axes[1].axvspan(1, 11-6, color='black', alpha=1, zorder=2)
    plt.tight_layout()
    plt.savefig(f'{path}/{row["CXO Name"]}.png')