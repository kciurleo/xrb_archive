import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates

#table=pd.read_csv('/Users/katieciurleo/Downloads/yalestuff/psf_fluxes.csv', low_memory=False)
#table=pd.read_csv('/home/kmc249/Downloads/newest_psf_fluxes_neighborhood_2008_apphot.csv', low_memory=False)
#table=pd.read_csv('/home/kmc249/Downloads/1m_psf_fluxes_all.csv', low_memory=False)
table=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_smaller_ap.csv', low_memory=False)
#table=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_08.csv', low_memory=False)
#extratable=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv', low_memory=False)
#extratable2=pd.read_csv('/home/kmc249/Downloads/psf_fluxes_2011.csv', low_memory=False)
#table = pd.concat([table, extratable], ignore_index=True)
#table = pd.concat([table, extratable2], ignore_index=True)
table['nice time'] = pd.to_datetime(table['time'])
#table2=pd.read_csv('/home/kmc249/Downloads/1m_wideR_psf_fluxes.csv', low_memory=False)
table2=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv', low_memory=False)
#table2=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_smaller_ap.csv', low_memory=False)
table2['nice time'] = pd.to_datetime(table2['time'])
#table2= table2[table2['nice time'].dt.year == 2008]
standards=pd.read_csv('/home/kmc249/Downloads/BEST_ens_stds_info.csv')
#table=table2
print(table.head)


def f(x, a, c):
    return a*np.log10(x)+c

fig, axes = plt.subplots(figsize=(8, 8))
xdata, ydata=[],[]
threshold = 1e4
bad_ids = []
mag_dict = {} 
for e in table.columns:
    if e not in ['nice time', 'time', 'filename', '1320', '413']:#, '1418','1069','1105','1320', 'a','b','c','d',]:
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes


        x = np.std(flux_safe)
        #if x < threshold:
		#if e not in ['aql', 'neighbor']:
		#	bad_ids.append(e)
        y = -2.5 * np.log10(np.nanmean(flux_safe))
        mag_dict[e] = y
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
#axes.set_yscale('log')
popt, pcov = curve_fit(f, np.array(xdata), np.array(ydata))
x_arr=np.linspace(np.min(xdata), np.max(xdata),150)
axes.plot( f(x_arr, *popt),x_arr, 'g--')
axes.invert_xaxis()

#plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_ensemble_variability.png', dpi=250)
plt.show(block=False)
print(list(map(int, bad_ids)))


#mean aql mag
mean_aql_mag = mag_dict['aql']
distances = []

for k, v in mag_dict.items():
    if k != 'aql':
        dist = abs(v - mean_aql_mag)
        distances.append((k, dist, v))
        
# sort by closeness
distances.sort(key=lambda x: x[1])

for k, dist, mag in distances[:4]:
    print(f"{k}: Δmag={dist:.4f}, mag={mag:.3f}")
    
#%%

#print(askjhd)

fig, axes = plt.subplots(figsize=(8, 8))
xdata3=[]
ydata3=[]
badlist=[]
for e in table.columns:
    if  e not in ['nice time','time', 'filename', '1320','a','b','c','d',]:
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
#plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_ensemble_to_stds_psf.png', dpi=250)
plt.show()
print('badlist:',badlist)

fig, axes = plt.subplots(figsize=(8, 8))
for e in table.columns:
    if  e not in ['nice time','time', 'filename', '1320','a','b','c','d',]:
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
        y=row['r'].iloc[0]-(slope*(-2.5 * np.log10(np.nanmean(flux_safe)))+intercept)
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
#plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_resids_stds_color.png', dpi=250)
plt.show()

print('badlist: ',badlist)


exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','a','b','c','d','1418','1069','1105', '1320', 'aql mag','ave mag']

to_sum = [table[name] for name in table.columns if name not in exclude_cols]
print(to_sum)
avg=np.nanmean(to_sum)

#getting ave ens magnitude overall
total_avg=-2.5*np.log10(avg)
offset=total_avg+intercept

table['aql mag'] = np.nan       # pre-create column
table['ave mag'] = np.nan      # if needed for table2
'''
plt.figure(figsize=(12,3))
print('to sum:', [name for name in table.columns if name not in exclude_cols])
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
            mags = slope*(-2.5 * np.log10(flux) +2.5*np.log10(avg))+offset
            table.at[id, 'aql mag']=mags
            ave=slope*(-2.5*np.log10(avg))+intercept
            table.at[id, 'ave mag']=ave
            h2=plt.scatter(row['nice time'], mags+12.25, marker='.', color='k',label=f'{name}', s=15)
            #h3=plt.scatter(row['nice time'], ave, marker='.', color='grey', s=15)
#handles = [h2, h3]
#labels = ['aql', 'ens (offset)']
#plt.legend(handles=handles, labels=labels)
plt.ylabel('Pan-STARRS r')
#plt.ylim(20,16.5)
plt.gca().invert_yaxis()


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
plt.savefig('/home/kmc249/Downloads/aql_1m_lc_psf_try1.png', dpi=250)
plt.show()

eids= [431, 244, 214, 522, 1199, 545, 948, 271, 1065, 1423, 1115, 679, 295, 1081, 1434, 1416, 318, 1140, 397, 1413, 269, 1476, 983, 659, 1146, 670, 376, 784, 1407, 1169, 1458, 1086, 566, 158, 729, 235, 1207, 783, 505, 482, 1337, 812, 1234, 996, 268, 1243, 290, 794, 1160, 1195, 1490, 213, 1053, 759, 1379, 1006, 1280, 704, 1187, 1069, 173, 1039, 744, 514, 153, 1192, 786, 1046, 751, 982, 713, 381, 1320, 1099, 1263, 1167, 1113, 160, 582, 461, 533, 134, 613, 1215, 825, 1460, 1433, 757, 307, 1451, 758, 493, 597, 1477, 1085, 403, 1038, 67, 479, 1341, 1344, 1305, 80, 863, 82, 66, 104, 895, 55, 890, 1340, 163, 1380, 178, 224, 1345, 116, 234, 139, 218, 187, 215, 155, 226, 120, 209, 1382, 309, 1058, 399, 1404, 374, 262, 1007, 1414, 1399, 258, 312, 1385, 1415, 300, 260, 1042, 345, 288, 377, 1388, 1072, 371, 331, 1021, 369, 1023, 404, 395, 1109, 485, 492, 467, 1435, 418, 451, 413, 1094, 503, 445, 1418, 417, 1105, 410, 499, 1104, 1419, 506, 621, 1470, 639, 1161, 1457, 525, 1143, 1440, 1467, 657, 668, 641, 1139, 1471, 534, 695, 1165, 1459, 1141, 558, 530, 1145, 622, 1191, 1132, 1443, 792, 1238, 756, 1502, 1241, 1225, 1504, 1261, 703, 1506, 707, 681, 1489, 1509, 1483, 761, 1208, 1242, 781, 1512, 1211, 1519, 813, 800, 820]

#double plot
plt.figure(figsize=(4,3))
plt.hist(table['ave mag'], bins=45)
plt.xlabel('r mag')
plt.gca().invert_xaxis()
plt.show()
vals = table['ave mag'].values
vals = vals[np.isfinite(vals)]  # drop NaNs

mean = np.mean(vals)
std = np.std(vals, ddof=1)      # sample standard deviation
sdom = std / np.sqrt(len(vals)) # standard deviation of the mean

print(f"mean = {mean:.5f}")
print(f"std  = {std:.5f}")
print(f"sdom = {sdom:.5f}")
'''

exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','a','b','c','d','1418','1069','1105', '1320']
table['aql mag'] = np.nan       # pre-create column
table2['aql mag'] = np.nan      # if needed for table2
def process_and_plot(table, label='aql', color='k'):

    # ---- Compute ensemble zero-point offset for this table ----
    to_sum = [table[name] for name in table.columns if name not in exclude_cols]
    avg = np.nanmean(to_sum)
    total_avg = -2.5*np.log10(avg)
    offset = total_avg + intercept

    # ---- Plot each row ----
    for id, row in table.iterrows():

        # ensemble mean
        to_sum = [row[name] for name in table.columns if name not in exclude_cols]
        avg = np.nanmean(to_sum)

        # science star
        flux = row['aql']
        if flux <= 0 or np.isnan(flux):
            continue

        mags = -2.5*np.log10(flux) + 2.5*np.log10(avg) + offset
        table.at[id, 'aql mag']=mags
        plt.scatter(row['nice time'], mags, marker='.', color=color, s=15, label=label)

#%%
# ---------------- MAIN PLOT ----------------
merged = table.merge(table2, on='nice time', suffixes=('_ap', '_psf'))
merged['resid'] = merged['aql mag_ap'] - merged['aql mag_psf']

fig, (ax1, ax_resid) = plt.subplots(
    2, 1, figsize=(20,5),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)
plt.sca(ax1)
plt.sca(ax1)

process_and_plot(table, label='AP', color='k')
process_and_plot(table2, label='PSF', color='red')

handles, labels = ax1.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax1.legend(unique.values(), unique.keys())

ax1.set_ylabel('Pan-STARRS r')
ax1.set_ylim(18.5, 16)
#plt.gca().invert_yaxis()

ax_resid.scatter(
    merged['nice time'],
    merged['resid'],
    color='blue',
    s=10, label='AP - PSF'
)

ax_resid.axhline(0, color='gray', linestyle='--')
ax_resid.set_ylabel('Resids')
ax_resid.set_xlabel('Date')
ax_resid.set_ylim(0.5, -0.5)
ax_resid.legend()

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())

tick_locs = ax1.get_xticks()
tick_dates = mdates.num2date(tick_locs)
tick_mjds = Time(tick_dates).mjd

ax2.set_xticks(tick_locs)
ax2.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])

ax2.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('bottom')
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.show()
print(table[['nice time', 'aql mag']].head())
print(table2[['nice time', 'aql mag']].head())
#table[['nice time', 'aql mag']].to_csv('/home/kmc249/Downloads/rough_aql_r.csv', index=False)
#table2[['nice time', 'aql mag']].to_csv('/home/kmc249/Downloads/rough_aql_wide_r.csv', index=False)
print(aksjhdasjh)
#%%
#periodogramming things
#table=table.loc[(table['aql mag']>16.5) & (table['aql mag']<19.2)]
baseline=table2['nice time'].max()-table2['nice time'].min()
base_days=baseline.total_seconds() / 3600 /24
print(base_days)

#folded??
P = 0.789498  # period in days
times = Time(table['nice time']).mjd
t0 = times.min()
phase = ((times - t0) / P) % 1
phase = (phase + 0.1) % 1
table['their phase']=phase

##periodograms
min_frequency = 24/19.5
max_frequency = 24/18.5

deltaf=P/base_days/4
print('DELTA F:', deltaf)
print(np.abs(min_frequency-max_frequency)/10000)

frequency = np.arange(min_frequency, max_frequency, deltaf)
from astropy.timeseries import LombScargle
fall, pall = LombScargle(times, table['aql mag']-np.nanmean(table['aql mag'])).autopower()
power = LombScargle(times, table['aql mag']-np.nanmean(table['aql mag'])).power(frequency)

# Convert frequency to period in hours
period_hours = 24 / frequency
sorted_idx = np.argsort(period_hours)
period_hours_sorted = period_hours[sorted_idx]
power_sorted = power[sorted_idx]

# Plot periodogram in period units
plt.figure(figsize=(8,4))
plt.plot(period_hours_sorted, power_sorted)
plt.xlabel('Period (hours)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.show(block=False)

fig, ax = plt.subplots()
ax.plot(frequency, power)
plt.show(block=False)

fig, ax = plt.subplots()
ax.plot(fall, pall)
plt.show(block=False)

best_frequency = frequency[np.argmax(power)]
P2 = 1 / best_frequency
best_period_hours = P2 * 24
print(best_period_hours)

phase2 = ((times - t0) / P2) % 1
table['our phase']=phase2

# Number of bins
nbins = 25
bins = np.linspace(0, 1, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Assign each phase to a bin
table['our phase bin'] = pd.cut(table['our phase'], bins=bins, include_lowest=True, labels=bin_centers)
table['their phase bin'] = pd.cut(table['their phase'], bins=bins, include_lowest=True, labels=bin_centers)

# Compute mean and std per bin
binned = table.groupby('our phase bin')['aql mag'].agg(['mean','std']).reset_index()
binned_them = table.groupby('their phase bin')['aql mag'].agg(['mean','std']).reset_index()

# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase2, table['aql mag'], s=15, color='gray', label='Data')
plt.scatter(phase2 + 1, table['aql mag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=binned['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=binned['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('Pan-STARRS r')
#plt.ylim(18.8, 18.1)
plt.gca().invert_yaxis()
plt.title(f'Our Period: {best_period_hours} hrs')
plt.legend()
plt.tight_layout()
plt.show(block=False)


# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase, table['aql mag'], s=15, color='gray', label='Data')
plt.scatter(phase + 1, table['aql mag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('Pan-STARRS r')
#plt.ylim(18.8, 18.1)
plt.gca().invert_yaxis()
plt.legend()
plt.title(f'Their Period: {P*24} hrs')
plt.tight_layout()
plt.show()


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Ellipsoidal binary model ---

def binary_model(phi, m0, A1, A2, phi0):
    return m0 + A1 * np.sin(2*np.pi*(phi - phi0)) + A2 * np.sin(4*np.pi*(phi - phi0))



# --- Extract arrays ---
phi = table['our phase'].values
mag = table['aql mag'].values

# Remove NaNs
mask = ~np.isnan(phi) & ~np.isnan(mag)
phi = phi[mask]
mag = mag[mask]

# --- Fit the model ---
# initial guess for phase shift: 0
p0 = [np.mean(mag), 0.1, 0.1, 0.0]
params, cov = curve_fit(binary_model, phi, mag, p0=p0)
m0, A1, A2, phi0 = params
print(f"m0={m0:.4f}, A1={A1:.4f}, A2={A2:.4f}, phase shift={phi0:.4f}")


# --- Model curve for plotting ---
phi_fit = np.linspace(0, 1, 500)
mag_fit = binary_model(phi_fit, *params)

# --- Plot ---
plt.figure(figsize=(10,5))

# Data
plt.scatter(phi, mag, s=15, color='gray', alpha=0.7, label='Data')
plt.scatter(phi+1, mag, s=15, color='gray', alpha=0.4)

# Binned averages (already computed earlier)
plt.errorbar(binned['our phase bin'].astype(float),
             binned['mean'], yerr=binned['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned['our phase bin'].astype(float)+1,
             binned['mean'], yerr=binned['std'],
             fmt='o', color='red', alpha=0.5)

# Best-fit model
plt.plot(phi_fit, mag_fit, color='blue', linewidth=2, label='Binary Model')
plt.plot(phi_fit+1, mag_fit, color='blue', linewidth=2, alpha=0.5)

# Labels
plt.gca().invert_yaxis()
plt.xlabel("Orbital Phase")
plt.ylabel("Pan-STARRS r")
plt.title("Best-Fit Ellipsoidal Binary Light Curve")
plt.legend()
plt.tight_layout()
plt.show()

perr = np.sqrt(np.diag(cov))
m0_err, A1_err, A2_err, phierr = perr

print("Uncertainties:")
print(f"m0_err = {m0_err}")
print(f"A1_err = {A1_err}")
print(f"A2_err = {A2_err}")
print(f"phierr = {phierr}")
