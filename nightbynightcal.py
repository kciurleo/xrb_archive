#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:39:59 2025

@author: kmc249
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
#2009 and 2019 isn't real, actually edit these

g=pd.DataFrame(columns=['START', 'END'])
g['START']=['1998-06-04', '1999-01-01','2000-01-01', '2001-01-01', '2001-07-01','2002-01-01','2004-01-01', '2003-02-03', '2003-07-28', '2009-09-01', '2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01']

g['END']=['1998-12-31','1999-12-31','2000-12-31', '2001-08-22','2001-12-31', '2002-09-26', '2004-09-23', '2003-04-25', '2004-01-01', '2009-12-31', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31', '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-09-01']

g['color']=['green', 'green', 'green', 'green','green','green', 'yellow','yellow', 'yellow','green','green','green','green','green','green','green','green','green','green','green',]

g['START']=pd.to_datetime(g['START'], errors='coerce')
g['END']=pd.to_datetime(g['END'], errors='coerce')

#set up years array
years=np.arange(1998,2020,1)


f, a = plt.subplots(11, 2, figsize=(10,8))
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=g.loc[g['START'].dt.year==year]
    for jd, row in yrtable.iterrows():
        start = row['START']
        end = row['END']

        axes[id].barh(y=0, width=(end - start).days, left=start, height=0.4, color=row['color'])
        #account for span crossing year
        if end.year==year+1:
            axes[id+1].barh(y=0, width=(end - start).days, left=start, height=0.4, color=row['color'])


    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticklabels([]) 
    axes[id].set_yticks([])
    
    if year!=2008 and year!=2019:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

#save as a file
plt.suptitle('Night-by-Night')
plt.tight_layout()

plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/night_by_night_access.png')
plt.show()