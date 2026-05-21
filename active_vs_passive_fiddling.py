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


for key in data:
    df = data[key]['df']
    if key=='J SMARTS':
        nameband='J'
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
intervals = list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))

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
'''
#Resaving files because I suck
for key in data:
    savetable=data[key]['df']
    band=key.split(' ')[0]
    header = f"# MJD corrected {band} MAG corrected uncertainty upperlimitflag {band} MAG uncertainty error_flag"
    if not 'LCO' in key:
        tosave=savetable[['MJD', 'mag_shifted', 'mag_corr_err', 'flag', 'mag', 'mag_err', 'error_flag']]
        with open(f'/home/kmc249/Downloads/take3/{band}_SMARTS_corrected.txt', "w") as file:
            file.write(header + "\n")
            
            tosave.to_csv(
                file,
                sep=" ",
                index=False,
                header=False,
            )
        continue

    tosave=savetable[['MJD', 'mag_shifted', 'mag_corr_err', 'flag', 'mag', 'mag_err', 'error_flag']]
    with open(f'/home/kmc249/Downloads/take3/{band}_usable_full_corrected.txt', "w") as file:
        file.write(header + "\n")
        
        tosave.to_csv(
            file,
            sep=" ",
            index=False,
            header=False,
        )
'''
#%%

#epochs, I think based on all times

def get_epoch_boundaries(all_mjd, gap=42):
    all_mjd = np.sort(np.unique(all_mjd))

    boundaries = [all_mjd[0]]

    for i in range(1, len(all_mjd)):
        if all_mjd[i] - all_mjd[i-1] >= gap:
            boundaries.append(all_mjd[i])

    boundaries.append(all_mjd[-1] + 1)

    return boundaries

def assign_epochs_from_boundaries(df, boundaries):
    df = df.copy()

    df["epoch"] = np.digitize(df["MJD"], boundaries) - 1
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
        "difference": local-global_med
    })


#%%

#null hypothesis is that for a given epoch (of a given size), its local median is explainable by random choice
#hypothesis other than that is that the local median is statistically significant

def get_pvalue(key,a=0.05, n_iter=10000):
    #for a given band, let's randomly sample the same size as its epochs and
    #get median values. Run a monte-carlo P test for the null hypothesis and 
    #return an array of p values for the epochs.
    df=data[key]['quiescence']

    grouped = df.groupby("epoch")

    all_vals = df["mag_shifted"].values
    results = []
    #For each differently sized epoch:
    for epoch, group in grouped:
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
        
        #monte carlo p-value?
        center = np.median(all_vals) 

        p_value = (np.sum(np.abs(fake_medians - center) >= abs(obs_med - center)) + 1) / (n_iter + 1)
        if p_value<a:
            sig='significant'
        else:
            sig='not significant'
        results.append({
            "epoch": epoch,
            "p_value": p_value,
            "n": n,
            "obs_median": obs_med,
            "sig": sig
        })
        try:
            plt.figure(figsize=(8,4))
            plt.hist(fake_medians, bins=50)
            plt.title(f'{key}, epoch {epoch}')
            plt.savefig('/home/kmc249/Downloads/simulated_histograms/f{key}_{epoch}.png')
            plt.show()
        except:
            continue

    return pd.DataFrame(results)



#%%
#find the epochs and compute some medians
all_mjd = pd.concat([data[k]["quiescence"][["MJD"]] for k in data])["MJD"].values
boundaries = get_epoch_boundaries(all_mjd, gap=42)
for key in data:
    df = data[key]['quiescence']

    # 1. assign epochs
    df = assign_epochs_from_boundaries(df, boundaries)
    data[key]['quiescence'] = df

    # 2. local epoch medians
    epoch_medians = compute_epoch_medians(df)

    # 3. classification
    epoch_table = classify_epochs(epoch_medians, data[key]['global med q mag'])

    data[key]['epoch_table'] = epoch_table
    df = df.merge(data[key]['epoch_table'][["epoch", "state"]], on="epoch", how="left")


#check to see if epochs agree
epoch_states = []
for key in data:
    data[key]["epoch_pvalues"] = get_pvalue(key, n_iter=10000)

    data[key]['epoch_table'] = data[key]['epoch_table'].merge(
        data[key]["epoch_pvalues"][["epoch", "p_value", "sig"]],
        on="epoch",
        how="left"
    )

    tmp = data[key]["epoch_table"].copy()
    tmp["band"] = key
    epoch_states.append(tmp)

epoch_states = pd.concat(epoch_states, ignore_index=True)
comparison = epoch_states.pivot_table(
    index="epoch",
    columns="band",
    values="state",
    aggfunc="first"
)
def row_agreement(row):
    vals = row.dropna().values

    # remove "none"
    vals = vals[vals != "none"]

    if len(vals) == 0:
        return "no_data"

    if np.all(vals == vals[0]):
        return "agree"

    return "disagree"

comparison["agreement"] = comparison.apply(row_agreement, axis=1)
def agreement_fraction(row):
    vals = [v for v in row.values if v in ["active", "passive"]]

    if len(vals) == 0:
        return np.nan

    most_common = max(set(vals), key=vals.count)
    return vals.count(most_common) / len(vals)

comparison["disagreement_score"] = comparison.apply(agreement_fraction, axis=1)

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
    plt.show()

#%%
for key in data:
    ks_test(key,a=0.05, n_iter=10000)

#%%
print(aksjhdsj)
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
    
    # ---- shaded epoch regions ----
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        epoch_id = i

        match = comparison.loc[epoch_id, "agreement"] if epoch_id in comparison.index else "no_data"

        if match == "agree":
            color = "green"
        elif match == "disagree":
            color = "red"
        else:
            color = "gray"

        ax.axvspan(start, end, color=color, alpha=0.2)

    # ---- scatter plot colored by state ----
    for state, color in state_colors.items():
        sub = df[df["state"] == state]
        ax.scatter(sub["MJD"], sub["mag_shifted"],
                   s=10, color=color, label=state, alpha=0.8)

    # ---- vertical epoch boundaries ----
    for x in boundaries:
        ax.axvline(x, color="k", alpha=0.3, linewidth=1)


    ax.set_title(key)
    ax.set_ylabel("Aql mag")
    ax.invert_yaxis()
    
    ax.axhline(item['global med q mag'], linewidth=1, color='k', linestyle='--',label='Global median')

axes[-1].set_xlabel("MJD")

handles, labels = axes[0].get_legend_handles_labels()
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
    "significant": '*',
    "not significant": '.',
    
}
fig, axes = plt.subplots(len(data), 1, figsize=(19, 3 * len(data)), sharex=True)

if len(data) == 1:
    axes = [axes]

for ax, (key, item) in zip(axes, data.items()):

    df = item["plot_df"]
    print(df.columns)
    
    # ---- shaded epoch regions (per-band significance) ----
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
    
        epoch_id = i
    
        sub_epoch = df[df["epoch"] == epoch_id]
    
        if len(sub_epoch) == 0:
            color = "gray"
        else:
            # take dominant sig in epoch (or median decision)
            sig_mode = sub_epoch["sig"].mode()
            sig_mode = sig_mode.iloc[0] if len(sig_mode) > 0 else "not significant"
    
            if sig_mode == "significant":
                color = "green"
            else:
                color = "red"
    
        ax.axvspan(start, end, color=color, alpha=0.2)

    # ---- scatter plot colored by state ----
    for state, color in state_colors.items():
        sub = df[df["state"] == state]
        for sig, marker in sig_markers.items():
            newsub=sub[sub["sig"]==sig]
            ax.scatter(newsub["MJD"], newsub["mag_shifted"],
                       s=10, color=color, marker=marker, label=state, alpha=0.8)

    # ---- vertical epoch boundaries ----
    for x in boundaries:
        ax.axvline(x, color="k", alpha=0.3, linewidth=1)


    ax.set_title(key)
    ax.set_ylabel("Aql mag")
    ax.invert_yaxis()
    
    ax.axhline(item['global med q mag'], linewidth=1, color='k', linestyle='--',label='Global median')

axes[-1].set_xlabel("MJD")

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend()

plt.tight_layout()
plt.show()