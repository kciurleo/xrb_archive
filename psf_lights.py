import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates

#table=pd.read_csv('/Users/katieciurleo/Downloads/yalestuff/psf_fluxes.csv', low_memory=False)
table=pd.read_csv('/Users/katieciurleo/Downloads/psf_fluxes.csv', low_memory=False)
table['nice time'] = pd.to_datetime(table['time'])

standards=pd.read_csv('/Users/katieciurleo/Downloads/yalestuff/BEST_ens_stds_info.csv')

def f(x, a, c):
    return a*np.log10(x)+c

fig, axes = plt.subplots(figsize=(8, 8))
xdata, ydata=[],[]

for e in table.columns:
    if e not in ['nice time', 'time', 'filename', '1418','1069','1105','1320']:
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes

        x = np.std(flux_safe)
        y = -2.5 * np.log10(np.nanmean(flux_safe))
        if e not in ['aql']:
            xdata.append(x)
            ydata.append(y)
        axes.scatter(y, x)
        if e == 'aql': 
            a=1
        elif e == 'neighbor':
            a=1
        else:
            a=0.2
        axes.annotate(
            str(e),
            (y, x),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color='red',
            alpha=a

        )


axes.set_ylabel('std of flux')
axes.set_xlabel('mag')

popt, pcov = curve_fit(f, np.array(xdata), np.array(ydata))
x_arr=np.linspace(np.min(xdata), np.max(xdata),150)
axes.plot( f(x_arr, *popt),x_arr, 'g--')
axes.invert_xaxis()
plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_ensemble_variability.png', dpi=250)
plt.show(block=False)



fig, axes = plt.subplots(figsize=(8, 8))
xdata3=[]
ydata3=[]
badlist=[]
for e in table.columns:
    if  e not in ['nice time','time', 'filename', '1320']:
        try:
            row=standards.loc[standards['num int']==int(e)]
        except:
            continue
        if len(row)<1:
            continue
        y=row['r'].iloc[0]
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes

        x = -2.5 * np.log10(np.nanmean(flux_safe))
        if x>-10:
            badlist.append(int(e))
        xdata3.append(x)
        ydata3.append(y)
        axes.scatter(x, y)
        axes.annotate(
            str(e),
            (x,y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color='red',
            alpha=0.5
        )

axes.set_xlabel('psf mag of standard stars')
axes.set_ylabel('panstarrs mag')
slope, intercept, r, p, se =linregress(xdata3, ydata3)
x3_arr=np.linspace(np.min(xdata3), np.max(xdata3))
axes.plot(x3_arr, slope*x3_arr+intercept, 'g--', label=f'y={np.round(slope,2)}x+{np.round(intercept, 2)}')
axes.invert_yaxis()
axes.invert_xaxis()
plt.legend()
plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_ensemble_to_stds_psf.png', dpi=250)
plt.show()


fig, axes = plt.subplots(figsize=(8, 8))
for e in table.columns:
    if  e not in ['nice time','time', 'filename', '1320']:
        try:
            row=standards.loc[standards['num int']==int(e)]
        except:
            continue
        if len(row)<1:
            continue
        x=row['g'].iloc[0]-row['r'].iloc[0]
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes

        #y = -2.5 * np.log10(np.nanmean(flux_safe))-row['r'].iloc[0]
        y=np.abs(row['r'].iloc[0]-(slope*(-2.5 * np.log10(np.nanmean(flux_safe)))+intercept))
        axes.scatter(x, y)
        axes.annotate(
            str(e),
            (y,x),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color='red',
            alpha=0.5
        )

axes.set_ylabel('resid (panstarrs r - linear model)')
axes.set_xlabel('panstarrs g-r (mag)')
plt.legend()
plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_resids_stds_color.png', dpi=250)
plt.show()

print('badlist: ',badlist)


plt.figure(figsize=(12,3))

exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','1418','1069','1105', '1320']

to_sum = [table[name] for name in table.columns if name not in exclude_cols]
avg=np.nanmean(to_sum)

#getting ave ens magnitude overall
total_avg=-2.5*np.log10(avg)
offset=total_avg+intercept


for id, row in table.iterrows():
    # average flux of comparison stars only
    to_sum = [row[name] for name in table.columns if name not in exclude_cols]
    avg = np.nanmean(to_sum)
    #h1=plt.scatter(row['nice time'], -2.5*np.log10(avg), s=15, color='gray',label='mean ens mag')

    for name in table.columns:
        if name  in ['aql']:#['nice time','time','filename']:
            flux = row[name]
            # skip non-positive fluxes
            if flux <= 0 or np.isnan(flux):
                continue
            mags = -2.5 * np.log10(flux) +2.5*np.log10(avg)+offset
            h2=plt.scatter(row['nice time'], mags, marker='.', color='k',label=f'{name}', s=15)
handles = [h2]
labels = ['aql']
#plt.legend(handles=handles, labels=labels)
plt.ylabel('Pan-STARRS r')
plt.ylim(20,15)


# --- Primary x-axis: date ---
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# --- Secondary x-axis (MJD), properly aligned ---
ax2 = ax1.twiny()  # create a second x-axis that shares the same y
ax2.set_xlim(ax1.get_xlim())  # align limits

# Convert tick locations to MJD
tick_locs = ax1.get_xticks()
tick_dates = mdates.num2date(tick_locs)
tick_mjds = Time(tick_dates).mjd

ax2.set_xticks(tick_locs)
ax2.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])

# Shift second axis downward for clarity (optional)
ax2.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('bottom')
plt.subplots_adjust(bottom=0.25)
ax1.xaxis.set_ticks_position('top')
plt.tight_layout()
plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_lc_psf_try1.png', dpi=250)
plt.show()

eids= [431, 244, 214, 522, 1199, 545, 948, 271, 1065, 1423, 1115, 679, 295, 1081, 1434, 1416, 318, 1140, 397, 1413, 269, 1476, 983, 659, 1146, 670, 376, 784, 1407, 1169, 1458, 1086, 566, 158, 729, 235, 1207, 783, 505, 482, 1337, 812, 1234, 996, 268, 1243, 290, 794, 1160, 1195, 1490, 213, 1053, 759, 1379, 1006, 1280, 704, 1187, 1069, 173, 1039, 744, 514, 153, 1192, 786, 1046, 751, 982, 713, 381, 1320, 1099, 1263, 1167, 1113, 160, 582, 461, 533, 134, 613, 1215, 825, 1460, 1433, 757, 307, 1451, 758, 493, 597, 1477, 1085, 403, 1038, 67, 479, 1341, 1344, 1305, 80, 863, 82, 66, 104, 895, 55, 890, 1340, 163, 1380, 178, 224, 1345, 116, 234, 139, 218, 187, 215, 155, 226, 120, 209, 1382, 309, 1058, 399, 1404, 374, 262, 1007, 1414, 1399, 258, 312, 1385, 1415, 300, 260, 1042, 345, 288, 377, 1388, 1072, 371, 331, 1021, 369, 1023, 404, 395, 1109, 485, 492, 467, 1435, 418, 451, 413, 1094, 503, 445, 1418, 417, 1105, 410, 499, 1104, 1419, 506, 621, 1470, 639, 1161, 1457, 525, 1143, 1440, 1467, 657, 668, 641, 1139, 1471, 534, 695, 1165, 1459, 1141, 558, 530, 1145, 622, 1191, 1132, 1443, 792, 1238, 756, 1502, 1241, 1225, 1504, 1261, 703, 1506, 707, 681, 1489, 1509, 1483, 761, 1208, 1242, 781, 1512, 1211, 1519, 813, 800, 820]

'''

big_table=pd.read_csv('/Users/katieciurleo/Downloads/yalestuff/testaqlphottable.csv')
grouped = big_table.groupby(['name'])

fig, axes = plt.subplots(figsize=(8, 8))
xdata4=[]
ydata4=[]
for e in table.columns:
    if e not in ['nice time', 'time', 'filename']:
        try:
            row=standards.loc[standards['num int']==int(e)]
        except:
            continue
        if len(row)<1:
            continue
        y=row['r'].iloc[0]
        try:
            x=grouped.get_group(e)['m'].mean()
        except:
            print('issue with',e)
            x=np.nan
        if e not in [71]:
            xdata4.append(x)
            ydata4.append(y)
        axes.scatter(x, y)
        axes.annotate(
            str(e),
            (x,y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color='red'
        )

axes.set_xlabel('ap phot')
axes.set_ylabel('std mag from table')
#axes.plot(x4_arr, slope2*x4_arr+intercept2, 'g--')
axes.invert_yaxis()
axes.invert_xaxis()
plt.show()

'''