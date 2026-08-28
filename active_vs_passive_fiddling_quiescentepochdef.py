#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:48:48 2026

@author: kmc249
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
#%%
#read in all the dfs

data = {}

data['B SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_B_corrected_lc_4_27.csv')
}

data['V SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_V_corrected_lc_4_27.csv')
}

data['R SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv')
}

data['I SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv')
}

data['J SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_J_corrected_lc.csv')
}

data['H SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_H_corrected_lc.csv')
}

data['K SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_K_corrected_lc.csv')
}


for key in data:
    df = data[key]['df']
    if key=='J SMARTS':
        nameband='J'
    elif key=='K SMARTS':
        nameband='K'
    elif key=='H SMARTS':
        nameband='H'
    else:
        nameband='R'
    if key=='V SMARTS':
        df['mag_shifted']=df[f'{nameband}mag_shifted']
    else:
        df['mag_shifted']=df[f'{nameband}mag_corr']
    df['mag_corr_err']=df[f'e_{nameband}mag_corr']
    df['mag_err']=df[f'e_{nameband}mag']
    df['mag']=df[f'{nameband}mag']
    
    df['flag']=np.nan
    df['error_flag']=np.nan

#assume we've gotten rid of the bad guys already.

def read_corrected_txt(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=[
            "MJD",
            "mag_corr",
            "mag_corr_err",
            "flag",
            "mag",
            "mag_err",
            "error_flag"
        ]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

def read_lco(banzai_path, orac_path):
    banzai = read_corrected_txt(banzai_path)
    orac   = read_corrected_txt(orac_path)
    return pd.concat([banzai, orac], ignore_index=True)


for band in ['V','gp','R','rp','ip']:
    data[f'{band} LCO']={'df':pd.read_csv(f'/home/kmc249/Downloads/{band}_LCO_shifted.csv')}

data['V LCO']['df']['mag_shifted']=data['V LCO']['df']['mag_corr']
data['V LCO']['df']['alt_mag_shifted']=data['V LCO']['df']['alt_mag_corr']

#only get stuff in quiescence

#Outburst list
full = pd.read_csv("/home/kmc249/Downloads/full_outbursts.csv")
mini = pd.read_csv("/home/kmc249/Downloads/mini_outbursts.csv")

#Mask out quiescence
intervals = list(zip(full["Start MJD"], full["End MJD"])) #+ \
            #list(zip(mini["Start MJD"], mini["End MJD"]))

def get_quiescent(df, intervals):
    mask = np.ones(len(df), dtype=bool)
    
    for start, end in intervals:
        mask &= ~((df["MJD"] >= start) & (df["MJD"] <= end))
    
    return df[mask].copy()

for key in data:
    df = data[key]['df']
    data[key]['quiescence'] = get_quiescent(df, intervals)
    data[key]['global med q mag'] = data[key]['quiescence']['mag_shifted'].median()
    data[key]['global mean q mag'] = data[key]['quiescence']['mag_shifted'].mean()
    print(key, data[key]['global mean q mag'])
    
    
    

#%%

#Resaving files because I suck
for key in data:
    savetable=data[key]['df']
    band=key.split(' ')[0]
    header = f"# MJD corrected {band} MAG corrected uncertainty upperlimitflag {band} MAG uncertainty error_flag"
    if not 'LCO' in key:
        tosave=savetable[['MJD', 'mag_shifted', 'mag_corr_err', 'flag', 'mag', 'mag_err', 'error_flag']]
        with open(f'/neta/xrb/AqlX-1/product/full_bands_subtracted/{band}_SMARTS_corrected.txt', "w") as file:
            file.write(header + "\n")
            
            tosave.to_csv(
                file,
                sep=" ",
                index=False,
                header=False,
            )
        continue

    tosave=savetable[['MJD', 'mag_shifted', 'mag_corr_err', 'flag', 'mag', 'mag_err', 'error_flag']]
    with open(f'/neta/xrb/AqlX-1/product/full_bands_subtracted/{band}_usable_full_corrected.txt', "w") as file:
        file.write(header + "\n")
        
        tosave.to_csv(
            file,
            sep=" ",
            index=False,
            header=False,
        )

#%%

#epochs, I think based on all times

# ============================================
# GLOBAL CROSS-BAND QUIESCENT EPOCH FRAMEWORK
# ============================================

def build_quiescent_epochs_from_outbursts(data, intervals):
    """
    Each epoch is a continuous quiescent interval:
    from end of one outburst to start of next outburst.

    This is physically defined, not sampling-defined.
    """

    # collect all observations across bands
    all_mjds = np.sort(np.unique(np.concatenate([
        data[k]["df"]["MJD"].values for k in data
    ])))

    # sort outburst intervals
    intervals = sorted(intervals)

    quiescent_starts = []
    quiescent_ends = []

    # start from beginning of dataset
    t_min = all_mjds[0]
    t_max = all_mjds[-1]

    # build gaps between outbursts
    prev_end = t_min

    for start, end in intervals:

        # quiescent segment before this outburst
        if start > prev_end:
            quiescent_starts.append(prev_end)
            quiescent_ends.append(start)

        prev_end = max(prev_end, end)

    # final quiescent stretch after last outburst
    if prev_end < t_max:
        quiescent_starts.append(prev_end)
        quiescent_ends.append(t_max)

    return np.array(quiescent_starts), np.array(quiescent_ends)


def assign_quiescent_epochs(df, starts, ends):
    """
    Assign each point to a physically defined quiescent epoch.
    """

    df = df.copy()
    df["epoch"] = -1

    for i, (s, e) in enumerate(zip(starts, ends)):
        mask = (df["MJD"] >= s) & (df["MJD"] < e)
        df.loc[mask, "epoch"] = i

    return df


def compute_epoch_medians(df):
    return df.groupby("epoch")["mag_shifted"].median()


def classify_epochs(epoch_medians, global_med):
    local = epoch_medians.values

    state = np.where(
        np.isnan(local),
        "none",
        np.where(local < global_med, "active", "passive")
    )

    return pd.DataFrame({
        "epoch": epoch_medians.index,
        "local_median": local,
        "state": state,
        "difference": local - global_med
    })

# ============================================
# PHYSICALLY DEFINED QUIESCENT EPOCHS
# ============================================

starts, ends = build_quiescent_epochs_from_outbursts(data, intervals)

print(f"Quiescent epochs (physical): {len(starts)}")

for key in data:

    df = data[key]["quiescence"].copy()

    df = assign_quiescent_epochs(df, starts, ends)

    data[key]["quiescence"] = df

    # compute stats per epoch
    epoch_medians = df.groupby("epoch")["mag_shifted"].median()

    epoch_table = classify_epochs(
        epoch_medians,
        data[key]["global med q mag"]
    )

    data[key]["epoch_table"] = epoch_table

    # merge back
    df = df.merge(
        epoch_table[["epoch", "state"]],
        on="epoch",
        how="left"
    )

    data[key]["quiescence"] = df

# ============================================
# MONTE CARLO SIGNIFICANCE TEST (UNCHANGED LOGIC)
# ============================================

def get_pvalue(key, a=0.05, n_iter=10000):

    df = data[key]["quiescence"]
    grouped = df.groupby("epoch")

    all_vals = df["mag_shifted"].values
    results = []

    for epoch, group in grouped:

        n = len(group)

        fake_medians = []
        for _ in range(n_iter):
            fake = np.random.choice(all_vals, size=n)
            fake_medians.append(np.median(fake))

        fake_medians = np.array(fake_medians)

        obs_med = np.median(group["mag_shifted"])
        center = np.median(all_vals)

        p_value = (np.sum(np.abs(fake_medians - center) >= abs(obs_med - center)) + 1) / (n_iter + 1)

        sig = "significant" if p_value < a else "not significant"

        results.append({
            "epoch": epoch,
            "p_value": p_value,
            "n": n,
            "obs_median": obs_med,
            "sig": sig
        })

    return pd.DataFrame(results)


# attach p-values
epoch_states = []

for key in data:

    data[key]["epoch_pvalues"] = get_pvalue(key, n_iter=10000)

    data[key]["epoch_table"] = data[key]["epoch_table"].merge(
        data[key]["epoch_pvalues"][["epoch", "p_value", "sig"]],
        on="epoch",
        how="left"
    )

    tmp = data[key]["epoch_table"].copy()
    tmp["band"] = key
    epoch_states.append(tmp)


epoch_states = pd.concat(epoch_states, ignore_index=True)


# ============================================
# CROSS-BAND COMPARISON
# ============================================

comparison = epoch_states.pivot_table(
    index="epoch",
    columns="band",
    values="state",
    aggfunc="first"
)


def row_agreement(row):
    vals = row.dropna().values
    vals = vals[vals != "none"]

    if len(vals) == 0:
        return "no_data"

    return "agree" if np.all(vals == vals[0]) else "disagree"


comparison["agreement"] = comparison.apply(row_agreement, axis=1)


def agreement_fraction(row):
    vals = [v for v in row.values if v in ["active", "passive"]]

    if len(vals) == 0:
        return np.nan

    most_common = max(set(vals), key=vals.count)
    return vals.count(most_common) / len(vals)


comparison["disagreement_score"] = comparison.apply(agreement_fraction, axis=1)


# merge into plotting frames
for key in data:

    df = data[key]["quiescence"].copy()

    df = df.merge(
        data[key]["epoch_table"][["epoch", "state", "p_value", "sig"]],
        on="epoch",
        how="left"
    )

    df = df.merge(
        comparison[["agreement"]],
        left_on="epoch",
        right_index=True,
        how="left"
    )

    data[key]["plot_df"] = df

#%%

#null hypothesis is that for a given epoch (of a given size), its local median is explainable by random choice
#hypothesis other than that is that the local median is statistically significant

def ks_test(key,a=0.05, n_iter=10000):
    #for a given band, let's randomly sample the same size as its epochs and
    #get median values. Run a monte-carlo P test for the null hypothesis and 
    #return an array of p values for the epochs.
    df=data[key]['plot_df']

    grouped = df.groupby("epoch")

    all_vals = df["mag_shifted"].values
    percentiles = []
    states=[]
    #For each differently sized epoch:
    for epoch, group in grouped:

        print(group.columns)
        n = len(group)
        #sample a bunch and find the medians of each sample
        fake_medians=[]
        for _ in range(n_iter):
            #simulated data
            fake = np.random.choice(all_vals, size=n)
            fake_med = np.median(fake)
            fake_medians.append(fake_med)
        #now for this epoch, we have a distribution of simulated medians
        fake_medians=np.array(fake_medians)
        #and the observed median
        obs_med = np.median(group["mag_shifted"])
        
        #how many simulated values are smaller?
        rank = np.sum(fake_medians < obs_med)  
        percentile = rank / len(fake_medians)
        
        percentiles.append(percentile)
        state = group["state"].mode().iloc[0] if "state" in group else "none"
        states.append(state)
    
    percentiles=np.array(percentiles)
    states=np.array(states)
    
    # split by state
    active = percentiles[states == "active"]
    passive = percentiles[states == "passive"]
    
    # x-grid
    x = np.linspace(0, 1, 200)
    
    # empirical CDFs
    y_active = np.array([np.mean(active <= xi) for xi in x])
    y_passive = np.array([np.mean(passive <= xi) for xi in x])
    y = np.array([np.mean(percentiles <= xi) for xi in x])
    
   
    
    # plot
    plt.figure(figsize=(6,5))
    
    plt.plot(x, y_active, color="blue", label="active")
    plt.plot(x, y_passive, color="orange", label="passive")
    plt.plot(x, y, label="All states", color='red')
    
    # ideal uniform line
    plt.plot([0, 1], [0, 1], '--', color='black', label="x=y")
    plt.title(key)
    plt.xlabel("x (percentile threshold)")
    plt.ylabel("Fraction of percentiles ≤ x")
    plt.legend()
    plt.savefig(f'/home/kmc249/Downloads/cdf_{key}.png')
    plt.show()

#%%
for key in data:
    ks_test(key,a=0.05, n_iter=10000)

#%%

#plotting business
state_colors = {
    "active": "blue",
    "passive": "orange",
    "none": "red"
}

fig, axes = plt.subplots(len(data), 1, figsize=(19, 3 * len(data)), sharex=True)

if len(data) == 1:
    axes = [axes]

for ax, (key, item) in zip(axes, data.items()):

    df = item["plot_df"]

    # ---- shaded physical quiescent epochs ----
    for i, (start, end) in enumerate(zip(starts, ends)):

        if i >= len(comparison):
            color = "gray"
        else:
            match = comparison["agreement"].iloc[i]

            if match == "agree":
                color = "green"
            elif match == "disagree":
                color = "red"
            else:
                color = "gray"

        ax.axvspan(start, end, color=color, alpha=0.2)

    # ---- scatter by state ----
    for state, color in state_colors.items():
        sub = df[df["state"] == state]

        ax.scatter(sub["MJD"], sub["mag_shifted"],
                   s=10, color=color, label=state, alpha=0.8)

    # ---- epoch boundaries (optional visual guide only) ----
    for x in starts:
        ax.axvline(x, color="k", alpha=0.2, linewidth=1)

    for x in ends:
        ax.axvline(x, color="k", alpha=0.2, linewidth=1)

    ax.set_title(key)
    ax.set_ylabel("Aql mag")
    ax.invert_yaxis()

    ax.axhline(item['global med q mag'],
               linewidth=1, color='k', linestyle='--')

axes[-1].set_xlabel("MJD")

axes[0].legend()
plt.tight_layout()
plt.show()

print('frac agrees')
print(comparison['agreement'].value_counts()['agree']/len(comparison))
print('total epochs:', len(comparison))

#%%
'''
#histograms
for key in data:
    df = data[key]["epoch_table"].copy()
    groups=df.groupby('state')
    x_min = df["difference"].min()
    x_max = df["difference"].max()
    bins = np.linspace(x_min, x_max, 61)
    plt.figure(figsize=(8,4))
    for state, group in df.groupby("state"):
        std=np.nanstd(group["difference"])
        med=group['difference'].median()
        if state=='none':
            continue
        if state=='active':
            c='blue'
            plt.axvline(-std, linestyle='--', color=c, label='std')
            plt.axvline(med, linestyle=':', color=c, label='median of active diff')
        elif state=='passive':
            c='orange'
            plt.axvline(std, linestyle='--', color=c, label='std')
            plt.axvline(med, linestyle=':', color=c, label='median of passive diff')
        plt.hist(
            group["difference"],
            bins=bins,
            alpha=0.5,
            label=state,
            density=True,
            color=c
        )
        
    plt.axvline(0, linestyle='-', color='k')

    plt.xlabel("magnitude difference")
    plt.gca().invert_xaxis()
    plt.ylabel("count")
    plt.title(f"{key}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
'''


#%%
sig_markers = {
    "significant": "*",
    "not significant": ".",
}

fig, axes = plt.subplots(len(data), 1, figsize=(19, 3 * len(data)), sharex=True)

if len(data) == 1:
    axes = [axes]

for ax, (key, item) in zip(axes, data.items()):

    df = item["plot_df"]

    # ---- shaded epoch significance ----
    for i, (start, end) in enumerate(zip(starts, ends)):

        sub_epoch = df[df["epoch"] == i]

        if len(sub_epoch) == 0:
            color = "gray"
        else:
            sig_mode = sub_epoch["sig"].mode()
            sig_mode = sig_mode.iloc[0] if len(sig_mode) > 0 else "not significant"

            color = "green" if sig_mode == "significant" else "red"

        ax.axvspan(start, end, color=color, alpha=0.2)

    # ---- scatter by state + significance ----
    for state, color in state_colors.items():
        sub = df[df["state"] == state]

        for sig, marker in sig_markers.items():
            newsub = sub[sub["sig"] == sig]

            ax.scatter(newsub["MJD"], newsub["mag_shifted"],
                       s=10, color=color, marker=marker, alpha=0.8)

    # ---- optional epoch boundaries ----
    for x in starts:
        ax.axvline(x, color="k", alpha=0.2, linewidth=1)

    for x in ends:
        ax.axvline(x, color="k", alpha=0.2, linewidth=1)

    ax.set_title(key)
    ax.set_ylabel("Aql mag")
    ax.invert_yaxis()

    ax.axhline(item['global med q mag'],
               linewidth=1, color='k', linestyle='--')

axes[-1].set_xlabel("MJD")

axes[0].legend()
plt.tight_layout()
plt.show()


#%%

for key in data:
    print('running ', key)
    if key!='R SMARTS':
        continue
    table=data[key]['plot_df']
    
    #get only passive
    table = table[table["state"] == 'passive']
    table['nice time']=Time(table['MJD'], format='mjd').to_datetime()
    maglabel=key
    magstring='mag_shifted'
    upper=19
    lower=22
    table=table.loc[(table[magstring]>upper) & (table[magstring]<lower)]
    
    
    #periodogramming things
    baseline=table['nice time'].max()-table['nice time'].min()
    base_days=baseline.total_seconds() / 3600 /24
    print(base_days)
    
    #folded??
    P = 0.789498  # period in days
    times = Time(table['nice time']).mjd
    t0 = times.min()
    phase = ((times - t0) / P) % 1
    table['their phase']=phase
    
    ##periodograms
    min_frequency = 24/18.96
    max_frequency = 24/18.94
    
    deltaf=P/base_days/4
    print('DELTA F:', deltaf)
    print(np.abs(min_frequency-max_frequency)/10000)
    
    frequency = np.linspace(min_frequency, max_frequency, 1000)#np.arange(min_frequency, max_frequency, deltaf)
    
    fall, pall = LombScargle(times, table[magstring]-np.nanmean(table[magstring])).autopower(maximum_frequency=2)
    power = LombScargle(times, table[magstring]-np.nanmean(table[magstring])).power(frequency)
    
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
    plt.axvline(x=P*24,alpha=0.5, color='red')
    plt.savefig(f'/home/kmc249/folded_curves/{key}_periodogram.png')
    plt.show(block=False)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(frequency, power)
    plt.show(block=False)
    
    # samme thing for fall pall
    pall_hours = 24 / fall
    sorted_idxall = np.argsort(pall_hours)
    period_hours_sortedall = pall_hours[sorted_idxall]
    power_sortedall = pall[sorted_idxall]
    
    # Plot periodogram in period units
    '''
    plt.figure(figsize=(8,4))
    plt.plot(period_hours_sortedall, power_sortedall)
    plt.xlabel('Period (hours)')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram (all freq, capped)')
    plt.axvline(x=P*24,alpha=0.5, color='red')
    plt.xlim(12, 45)
    plt.show(block=False)
    '''
    
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
    nbins = 16
    bins = np.linspace(0, 1, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Assign each phase to a bin
    table['our phase bin'] = pd.cut(table['our phase'], bins=bins, include_lowest=True, labels=bin_centers)
    table['their phase bin'] = pd.cut(table['their phase'], bins=bins, include_lowest=True, labels=bin_centers)
    
    # Compute mean and std per bin
    binned = table.groupby('our phase bin')[magstring].agg(['mean','std']).reset_index()
    binned_them = table.groupby('their phase bin')[magstring].agg(['mean','std']).reset_index()
    
    yerr_up_us = []
    yerr_down_us = []
    
    for center in binned['our phase bin']:
        # Get all points in this phase bin
        points = table.loc[table['our phase bin'] == center, magstring].values
        if len(points) == 0:
            yerr_up_us.append(0)
            yerr_down_us.append(0)
            continue
        
        mean_bin = np.mean(points)
        
        # Points above/below mean
        above = points[points > mean_bin]
        below = points[points < mean_bin]
        
        # 68% confidence ~ 1 sigma
        sigma_up = np.percentile(above, 68.3) - mean_bin if len(above) > 0 else 0
        sigma_down = mean_bin - np.percentile(below, 31.7) if len(below) > 0 else 0
        
        yerr_up_us.append(sigma_up)
        yerr_down_us.append(sigma_down)
    
    # Make 2×N array for asymmetric error bars
    yerr_asym_us = np.array([yerr_down_us, yerr_up_us])
    
    yerr_up_them = []
    yerr_down_them = []
    
    for center in binned['our phase bin']:
        # Get all points in this phase bin
        points = table.loc[table['our phase bin'] == center, magstring].values
        if len(points) == 0:
            yerr_up_them.append(0)
            yerr_down_them.append(0)
            continue
        
        mean_bin = np.mean(points)
        
        # Points above/below mean
        above = points[points > mean_bin]
        below = points[points < mean_bin]
        
        # 68% confidence ~ 1 sigma
        sigma_up = np.percentile(above, 68.3) - mean_bin if len(above) > 0 else 0
        sigma_down = mean_bin - np.percentile(below, 31.7) if len(below) > 0 else 0
        
        yerr_up_them.append(sigma_up)
        yerr_down_them.append(sigma_down)
    
    # Make 2×N array for asymmetric error bars
    yerr_asym_them = np.array([yerr_down_them, yerr_up_them])
    
    # Plot with asymmetric error bars
    plt.figure(figsize=(8,4))
    plt.scatter(phase2, table[magstring], s=15, color='gray', label='Data')
    plt.scatter(phase2 + 1, table[magstring], s=15, color='gray', alpha=0.5)
    plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=yerr_asym_us,
                 fmt='o', color='red', label='Binned Avg')
    plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=yerr_asym_us,
                 fmt='o', color='red', alpha=0.5)
    plt.xlabel('Orbital Phase')
    plt.ylabel(maglabel)
    plt.gca().invert_yaxis()
    plt.title(f'{magstring} Our Period: {best_period_hours} hrs')
    plt.legend()
    plt.tight_layout()
    
    plt.show(block=False)
    
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.scatter(phase, table[magstring], s=15, color='gray', label='Data')
    plt.scatter(phase + 1, table[magstring], s=15, color='gray', alpha=0.5)
    
    plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=yerr_asym_them,
                 fmt='o', color='red', label='Binned Avg')
    plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=yerr_asym_them,
                 fmt='o', color='red', alpha=0.5)
    
    plt.xlabel('Orbital Phase')
    plt.ylabel(maglabel)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'{magstring} Their Period: {P*24} hrs')
    plt.tight_layout()
    plt.show()
    
