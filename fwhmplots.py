#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 12:03:11 2025

@author: kmc249
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

fwhms=pd.read_csv('/home/kmc249/test_data/temp_aql_shifts.csv', low_memory=False)

bad_list=np.array(['130811.0067', '140529.0030','130504.0083','070524.0061','070809.0032','060710.0057', 
                   '070411.0052','070413.0070','030329.0212', '130504.0083','170730.0019', '171008.0030'
                   '160825.0042', '160904.0021', '170604.0070'])
bad_list='rccd'+bad_list+'.fits'

hm_list=np.array(['130820.0064', '130820.0064', '150408.0159','051010.0021','050517.0149','050719.0101',
                  '040531.0042','120906.0021','090928.0041','070729.0050','030321.0139','030307.0212'
                  ])
hm_list='rccd'+hm_list+'.fits'

bad=fwhms.loc[fwhms['filename'].isin(bad_list)]

hm=fwhms.loc[fwhms['filename'].isin(hm_list)]


plt.figure(figsize=(8,8))
plt.scatter(fwhms['xshift'], fwhms['yshift'])
plt.scatter(bad['xshift'], bad['yshift'], color='red')
plt.scatter(hm['xshift'], hm['yshift'], color='gold')

plt.show()