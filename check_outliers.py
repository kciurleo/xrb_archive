#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:42:17 2026

@author: kmc249
"""

import pandas as pd
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from astropy.time import Time

#df=pd.read_csv('/neta/xrb/AqlX-1/product/AqlX-1_R_corrected_lc_with_outliers.csv', low_memory=False)
#df=pd.read_csv('/neta/xrb/AqlX-1/product/AqlX-1_R_corrected_lc.csv', low_memory=False)
#df=pd.read_csv('/neta/xrb/AqlX-1/product/combo_subtracted_shifted/AqlX-1_R_corrected_lc_4_27.csv', low_memory=False)
df=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv', low_memory=False)
#df=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/AqlX-1_V_corrected_lc_4_27.csv', low_memory=False)


#cycle through. look at images. if bad, hold onto. 
bad_guys=[]


#For I:
bad_guys=['/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd100306.0108.fits','/neta/xrb/AqlX-1/1m/opt/rccd/I_trimmed/trim_r0818_1998.035.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/I_trimmed/trim_r0821_2298.044.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090325.0102.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090413.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090421.0107.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090426.0090.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090509.0080.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090519.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090521.0099.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090605.0055.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090608.0082.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090831.0105.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160816.0053.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160821.0029.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160821.0032.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160821.0035.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160901.0074.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160901.0077.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160912.0037.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160914.0005.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160928.0028.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd161003.0025.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd161010.0020.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170604.0145.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170608.0068.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170613.0144.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170614.0058.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170624.0046.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170712.0059.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170804.0062.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170822.0092.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170823.0112.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170902.0078.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170903.0062.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170918.0069.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170928.0039.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171006.0032.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171011.0016.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171015.0017.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171020.0015.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171022.0014.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171024.0014.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180330.0157.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180331.0166.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180418.0194.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180427.0165.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180428.0139.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180428.0142.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180501.0064.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180506.0165.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180508.0123.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180514.0175.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180527.0143.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180625.0071.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180626.0068.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180627.0067.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180628.0095.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180701.0095.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180705.0081.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180725.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180725.0101.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180730.0060.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180803.0072.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180811.0110.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180823.0118.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd181015.0026.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190421.0136.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190507.0169.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190515.0166.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190518.0115.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190518.0116.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190520.0135.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190521.0116.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190603.0195.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190622.0162.fits']
#For V
#bad_guys=['/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0713_1498.054.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0715_1698.048.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0717_1898.069.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0725_2698.051.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0730_3198.046.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0814_1598.038.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0818_1998.034.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0821_2298.043.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0906_0798.020.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1009_1098.009.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1011_1298.010.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1014_1598.011.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1022_2398.007.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1023_2498.007.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1024_2598.006.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1026_2798.006.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1028_2998.002.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1118_1998.002.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r1119_2098.002.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0528_2999.054.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0602_0399.080.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0622_2399.039.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_r0703_0499.046.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990728.0012.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990805.0016.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990825.0052.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990919.0028.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990920.0027.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990922.0026.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd990929.0018.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991003.0019.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991013.0015.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991016.0013.fits','/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991030.0008.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991108.0007.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991113.0002.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991114.0002.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd991115.0002.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd000512.0014.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd000515.0024.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd000516.0047.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/V_trimmed/trim_rccd000925.0002.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd060802.0114.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140712.0083.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140728.0041.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140807.0074.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140809.0099.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140810.0134.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140818.0076.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140820.0078.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140824.0083.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140906.0067.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140908.0069.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140910.0038.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140914.0046.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd140916.0059.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd141014.0029.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd141015.0033.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd141026.0031.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd141109.0003.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd150730.0063.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd151001.0038.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd151007.0042.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160804.0044.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160804.0047.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160805.0032.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160821.0030.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160908.0033.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160912.0038.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd160922.0020.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd161003.0026.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd161010.0021.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170514.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170604.0072.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170604.0146.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170605.0082.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170608.0069.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170613.0145.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170614.0128.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170706.0043.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170708.0050.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170712.0060.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170718.0117.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170804.0063.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170807.0101.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170823.0113.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170829.0065.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170902.0079.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170903.0063.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170905.0083.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170906.0073.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170927.0021.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd170928.0040.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd171001.0030.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd171003.0036.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd171006.0033.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd171007.0026.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd171015.0018.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd171018.0022.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180315.0151.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180420.0158.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180501.0065.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180508.0124.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180527.0165.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180623.0134.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180625.0072.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180626.0069.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180627.0068.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180628.0096.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180701.0096.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180705.0082.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180722.0074.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180725.0099.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180811.0111.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180823.0119.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180824.0130.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd180825.0140.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd190409.0150.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd190415.0139.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/V_trimmed/trim_rccd190421.0137.fits']

'''

skipto='1999-10-20 01:15:45.400'
skipover=True
for id, row in df.iterrows():
    if skipover:
        if row['nice time'] == skipto:
            skipover = False
        else:
            continue
    print(row['nice time'])
    filename=row['filename']
    try:
        im=fits.getdata(filename)
    except:
        print('NO FILE')
        continue
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(im)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(im, cmap='gray', origin='lower', norm=norm)
    plt.show()
    isbad=input('If bad type b')
    if 'b' in isbad:
        bad_guys.append(filename)
    if 'q' in isbad:
        break
'''
print(bad_guys)

#%%
df['nice time'] = pd.to_datetime(df['nice time'], errors='coerce')
bad_df=df.loc[df['filename'].isin(bad_guys)]

df = df.sort_values('nice time')

# difference with previous and next points
diff_prev = (df['Rmag_corr'] - df['Rmag_corr'].shift(1)).abs()
diff_next = (df['Rmag_corr'] - df['Rmag_corr'].shift(-1)).abs()

# flag points that differ by >2 from BOTH neighbors
blue_mask = (diff_prev > 2) & (diff_next > 2)
outliers = df[blue_mask]
outliers = outliers[~outliers['filename'].isin(bad_df['filename'])]

ymin = min(df['Rmag_corr'].min(), df['Rmag'].min())
ymax = max(df['Rmag_corr'].max(), df['Rmag'].max())

fig, axes = plt.subplots(
    7, 1, figsize=(16, 24),
    gridspec_kw={'height_ratios': [1,1,1,1,1,1,1]}
)

# get start and end times
t_start = df['nice time'].min()
t_end   = df['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=8)  # 4 chunks = 5 edges

# now split df into 4 chunks by time range
time_chunks = [df[(df['nice time'] >= edges[i]) & (df['nice time'] < edges[i+1])]
               for i in range(7)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = df[(df['nice time'] >= edges[-2]) & (df['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = edges[i]
    tmax = edges[i+1]
    
    # masks
    mask1 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax)
    mask2 = (bad_df['nice time'] >= tmin) & (bad_df['nice time'] <= tmax)
        
    mask3 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax) & blue_mask
    
    # --- MAIN LIGHT CURVE ---
    '''
    ax_main.errorbar(df.loc[mask1, 'nice time'],
                    df.loc[mask1,  'Rmag_corr'], yerr=np.abs(df.loc[mask1,  'e_Rmag_corr']), 
                    fmt='.', color='black', markersize=3, label='Corrected')
    '''
    ax_main.errorbar(df.loc[mask1, 'nice time'],
                    df.loc[mask1,  'Rmag_corr'], #yerr=df.loc[mask1,  'e_Rmag_corr'],
                    fmt='.', color='black', markersize=3, label='Corrected Rmag')
    ax_main.errorbar(df.loc[mask3, 'nice time'],
                     df.loc[mask3, 'Rmag_corr'],
                     #yerr=df.loc[mask3, 'e_Rmag_corr'],
                     fmt='.', color='cyan', markersize=4, label='Outliers')
    ax_main.errorbar(bad_df.loc[mask2, 'nice time'],
                    bad_df.loc[mask2,  'Rmag_corr'], #yerr=bad_df.loc[mask2,  'e_Rmag_corr'],
                    fmt='.', color='red', markersize=3, label='Excluded')
    
    ax_main.set_xlim(tmin, tmax)
    #ax_main.set_ylim(20, 15.3)
    ax_main.invert_yaxis()
    #ax_main.set_ylim(ymax, ymin) 
    
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')

plt.tight_layout()
#plt.savefig('/home/kmc249/Downloads/df.png', dpi=300)
plt.show()
#%%
'''
defobads=[]
for id, row in outliers.iterrows():
    print(row['nice time'])
    filename=row['filename']
    try:
        im=fits.getdata(filename)
    except:
        print('NO FILE')
        continue
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(im)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(im, cmap='gray', origin='lower', norm=norm)
    plt.show()
    isbad=input('If bad type b')
    if 'b' in isbad:
        defobads.append(filename)
    if 'q' in isbad:
        break
'''
#%%
real_bads=bad_guys#+list(outliers['filename'])
bad_df=df.loc[df['filename'].isin(real_bads)]
print(len(bad_df))
final_df=df[~df['filename'].isin(bad_df['filename'])]
print(len(final_df))
final_df
final_df=final_df.loc[final_df['Rmag_corr']<=24]
print(len(df))
#final_df.to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/AqlX-1_V_corrected_lc_4_27.csv')
#final_df.to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv')