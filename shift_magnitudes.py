#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:28:16 2026

@author: kmc249
"""

#bigplot
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

final_R=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv')
final_I=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv')
final_B=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_B_corrected_lc_4_27.csv')
final_V=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/AqlX-1_V_corrected_lc_4_27.csv')

#%%
for table in [final_R, final_V, final_I, final_B]:
    table['nice time'] = pd.to_datetime(table['nice time'], errors='coerce')
    
for table in [final_R, final_V, final_I, final_B]:
    table.dropna(subset=['nice time'], inplace=True)

fig, ax = plt.subplots(figsize=(14, 4))

ax.errorbar(final_R['nice time'], final_R['Rmag'],
            yerr=final_R['e_Rmag'],
            fmt='.', color='crimson', label='R', alpha=0.8)

ax.errorbar(final_V['nice time'], final_V['Rmag'],
            yerr=final_V['e_Rmag'],
            fmt='.', color='green', label='V', alpha=0.8)

ax.errorbar(final_I['nice time'], final_I['Rmag'],
            yerr=final_I['e_Rmag'],
            fmt='.', color='chocolate', label='I', alpha=0.8)

ax.errorbar(final_B['nice time'], final_B['Rmag'],
            yerr=final_B['e_Rmag'],
            fmt='.', color='blue', label='B', alpha=0.8)

ax.invert_yaxis()
ax.set_ylabel("Magnitude")
ax.set_xlabel("Time")
ax.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

bands = [
    ("B", final_B, "blue"),
    ("V", final_V, "green"),
    ("R", final_R, "crimson"),
    ("I", final_I, "chocolate"),
]

for ax, (label, df, color) in zip(axes, bands):
    ax.errorbar(
        df['nice time'],
        df['Rmag'],
        yerr=df['e_Rmag'],
        fmt='.',
        color=color,
        alpha=0.8,
        elinewidth=0.8,
        capsize=0
    )

    ax.set_ylabel(f"{label} mag")
    ax.invert_yaxis()
    ax.set_title(f"{label} band")

axes[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()


#%%
#reading in lco files
def read_uncorrected_txt(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["MJD", "mag", "mag_err", "flag"]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

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
            "alt_mag_corr"
        ]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

#%%
corrected = {}

corrected['R']=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv')
corrected['I']=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv')
#final_B=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_B_corrected_lc_4_27.csv')
corrected['V']=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/AqlX-1_V_corrected_lc_4_27.csv')

for band, df in corrected.items():
    df['nice time'] = pd.to_datetime(df['nice time'], errors='coerce')
    df.dropna(subset=['nice time'], inplace=True)

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True)#, sharey='row')

bands = [
    ("B", final_B, "blue"),
    ("V", final_V, "green"),
    ("R", final_R, "crimson"),
    ("I", final_I, "chocolate"),
]

for i, (label, df_old, color) in enumerate(bands):
    

    df_new = corrected.get(label)
        


    # ---- LEFT: original ----
    axes[i, 0].errorbar(
        df_old['nice time'],
        df_old['Rmag'],
        yerr=df_old['e_Rmag'],
        fmt='.',
        color=color,
        alpha=0.8
    )

    axes[i, 0].set_ylabel(f"{label} mag")

    if df_new is None:
        print(f"Skipping corrected {label} (not found)")
        continue
    # ---- RIGHT: corrected ----
    axes[i, 1].errorbar(
        df_new['nice time'],
        df_new['Rmag_corr'],
        yerr=0.,#df_new['e_Rmag_corr'],
        fmt='.',
        color=color,
        alpha=0.8
    )
# ---- invert ONCE per row ----
for i in range(4):
    ax_left = axes[i, 0]
    ax_right = axes[i, 1]

    # get current limits after plotting
    ymin, ymax = ax_left.get_ylim()
    #if i>0:
    #    ymin2, ymax = ax_right.get_ylim()


    # flip them manually (this is the key)
    ax_left.set_ylim(ymax+3, ymin)
    ax_right.set_ylim(ymax+3, ymin)
axes[0, 1].set_title("Corrected")
axes[0, 0].set_title("Original")
axes[-1, 0].set_xlabel("Time")
axes[-1, 1].set_xlabel("Time")

plt.tight_layout()
plt.show()


#%%

#R band
R_LCO_banzai = read_uncorrected_txt('/home/kmc249/Downloads/R_usable_banzai.txt')
R_LCO_banzai["from"]='banzai'  
R_LCO_orac = read_uncorrected_txt('/home/kmc249/Downloads/R_usable_orac.txt')
R_LCO_orac["from"]='orac'  
R_LCO=pd.concat([R_LCO_banzai, R_LCO_orac], ignore_index=True)
Rp_LCO = read_uncorrected_txt('/home/kmc249/Downloads/rp_usable_banzai.txt')     

#R band corr
R_LCO_banzai_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/R_usable_banzai_corrected.txt')  
R_LCO_banzai_corr["from"]='banzai'  
R_LCO_orac_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/R_usable_orac_corrected.txt')
R_LCO_orac_corr["from"]='orac'  
R_LCO_corr=pd.concat([R_LCO_banzai_corr, R_LCO_orac_corr], ignore_index=True)
Rp_LCO_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/rp_usable_banzai_corrected.txt') 

#ip band
ip_LCO_banzai = read_uncorrected_txt('/home/kmc249/Downloads/ip_usable_banzai.txt')  
ip_LCO_banzai["from"]='banzai'  
ip_LCO_orac = read_uncorrected_txt('/home/kmc249/Downloads/ip_usable_orac.txt')
ip_LCO_orac["from"]='orac'  
ip_LCO=pd.concat([ip_LCO_banzai, ip_LCO_orac], ignore_index=True)  

#ip band corr
ip_LCO_banzai_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/ip_usable_banzai_corrected.txt')  
ip_LCO_banzai_corr["from"]='banzai'  
ip_LCO_orac_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/ip_usable_orac_corrected.txt')
ip_LCO_orac_corr["from"]='orac'  
ip_LCO_corr=pd.concat([ip_LCO_banzai_corr, ip_LCO_orac_corr], ignore_index=True)


#v band
v_LCO_banzai = read_uncorrected_txt('/home/kmc249/Downloads/V_usable_banzai.txt')  
v_LCO_banzai["from"]='banzai'  
v_LCO_orac = read_uncorrected_txt('/home/kmc249/Downloads/V_usable_orac.txt')
v_LCO_orac["from"]='orac'  
v_LCO=pd.concat([v_LCO_banzai, v_LCO_orac], ignore_index=True)  

#v band corr
v_LCO_banzai_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/V_usable_banzai_corrected.txt')  
v_LCO_banzai_corr["from"]='banzai'  
v_LCO_orac_corr = read_corrected_txt('/neta/xrb/AqlX-1/product/just_subtracted_shifted/V_usable_orac_corrected.txt')
v_LCO_orac_corr["from"]='orac'  
v_LCO_corr=pd.concat([v_LCO_banzai_corr, v_LCO_orac_corr], ignore_index=True)

#%%
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

#find quiescence values for all tables
quiescent_tables = {}

for name, table in {
    "final_R": final_R,
    "final_V": final_V,
    "final_I": final_I
}.items():

    table = table.copy()
    table["MJD"] = Time(table["nice time"]).mjd

    quiescent_tables[name] = get_quiescent(table, intervals)

#find quiescence values for all tables which have MJD already
R_new=corrected.get("R")
V_new=corrected.get("V")
I_new=corrected.get("I")
extra_tables = {
    "R_new": R_new,
    "V_new": V_new,
    "I_new": I_new,
    "Rp_LCO": Rp_LCO,
    "Rp_LCO_corr": Rp_LCO_corr,
    "R_LCO": R_LCO,
    "R_LCO_corr": R_LCO_corr,
    "ip_LCO": ip_LCO,
    "ip_LCO_corr": ip_LCO_corr,
    "v_LCO": v_LCO,
    "v_LCO_corr": v_LCO_corr,
}

for name, table in extra_tables.items():
    quiescent_tables[name] = get_quiescent(table, intervals)
    
#%%

#get mean quiescent values and comparisons
#uncorrected
uncoorr_R_SMARTS=quiescent_tables['final_R']['Rmag'].mean()
uncoorr_R_LCO=quiescent_tables['R_LCO']['mag'].mean()
uncoorr_Rp_LCO=quiescent_tables['Rp_LCO']['mag'].mean()

uncoorr_V_SMARTS=quiescent_tables['final_V']['Rmag'].mean()
uncoorr_V_LCO=quiescent_tables['v_LCO']['mag'].mean()

uncoorr_I_SMARTS=quiescent_tables['final_I']['Rmag'].mean()
uncoorr_ip_LCO=quiescent_tables['ip_LCO']['mag'].mean()

#corrected
coorr_R_SMARTS=quiescent_tables['R_new']['Rmag'].mean()
coorr_R_LCO=quiescent_tables['R_LCO_corr']['mag'].mean()
coorr_Rp_LCO=quiescent_tables['Rp_LCO_corr']['mag'].mean()

coorr_V_SMARTS=quiescent_tables['V_new']['Rmag'].mean()
coorr_V_LCO=quiescent_tables['v_LCO_corr']['mag'].mean()

coorr_I_SMARTS=quiescent_tables['I_new']['Rmag'].mean()
coorr_ip_LCO=quiescent_tables['ip_LCO_corr']['mag'].mean()


#print differences
print('--- R BAND ---')
print(uncoorr_R_SMARTS)
print(uncoorr_R_LCO)
print(uncoorr_Rp_LCO)
print('us-lco')
print(uncoorr_R_SMARTS-uncoorr_R_LCO)
print(uncoorr_R_SMARTS-uncoorr_Rp_LCO)
print('rp-lco')
print(uncoorr_Rp_LCO-uncoorr_R_LCO)
print('')

print('--- I BAND ---')
print(uncoorr_I_SMARTS)
print(uncoorr_ip_LCO)
print('us-lco')
print(uncoorr_I_SMARTS-uncoorr_ip_LCO)
print('')

print('--- V BAND ---')
print(uncoorr_V_SMARTS)
print(uncoorr_V_LCO)
print('us-lco')
print(uncoorr_V_SMARTS-uncoorr_V_LCO)
print('')

print('corrected')

print('--- R BAND ---')
print(coorr_R_SMARTS)
print(coorr_R_LCO)
print(coorr_Rp_LCO)
print('us-lco')
print(coorr_R_SMARTS-coorr_R_LCO)
print(coorr_R_SMARTS-coorr_Rp_LCO)
print('rp-lco')
print(coorr_Rp_LCO-coorr_R_LCO)
print('')

print('--- I BAND ---')
print(coorr_I_SMARTS)
print(coorr_ip_LCO)
print('us-lco')
print(coorr_I_SMARTS-coorr_ip_LCO)
print('')

print('--- V BAND ---')
print(coorr_V_SMARTS)
print(coorr_V_LCO)
print('us-lco')
print(coorr_V_SMARTS-coorr_V_LCO)
print('')


#%%

#R band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

    # ---- LEFT: original ----
axes[0].errorbar(
    final_R['nice time'],
    final_R['Rmag'],
    yerr=final_R['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    R_LCO['nice time'],
    R_LCO['mag'],
    yerr=R_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[0].errorbar(
    Rp_LCO['nice time'],
    Rp_LCO['mag'],
    yerr=Rp_LCO['mag_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)


axes[0].set_ylabel("R mag")
axes[1].set_ylabel("R mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    R_new['nice time'],
    R_new['Rmag_corr'],
    yerr=R_new['e_Rmag_corr'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    R_LCO_corr['nice time'],
    R_LCO_corr['mag_corr'],
    yerr=R_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[1].errorbar(
    Rp_LCO_corr['nice time'],
    Rp_LCO_corr['mag_corr'],
    yerr=Rp_LCO_corr['mag_corr_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)

# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

axes[0].axhline(y=uncoorr_R_SMARTS, color='Red', linestyle='--', linewidth=3, alpha=0.5)
axes[0].axhline(y=uncoorr_R_LCO, color='black', linestyle='--', linewidth=3, alpha=0.5)
axes[0].axhline(y=uncoorr_Rp_LCO, color='gray', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_R_SMARTS, color='Red', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_R_LCO, color='black', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_Rp_LCO, color='gray', linestyle='--', linewidth=3, alpha=0.5)



plt.tight_layout()
plt.legend()
plt.show()


#%%


#V band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')
    # ---- LEFT: original ----
axes[0].errorbar(
    final_V['nice time'],
    final_V['Rmag'],
    yerr=final_V['e_Rmag'],
    fmt='.',
    color='green',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    v_LCO['nice time'],
    v_LCO['mag'],
    yerr=v_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO V'
)


axes[0].set_ylabel("V mag")
axes[1].set_ylabel("V mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    V_new['nice time'],
    V_new['Rmag_corr'],
    yerr=V_new['e_Rmag'],
    fmt='.',
    color='green',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    v_LCO_corr['nice time'],
    v_LCO_corr['mag_corr'],
    yerr=v_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO V'
)


# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
#ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax+8, ymin)
ax_right.set_ylim(ymax+8, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

axes[0].axhline(y=uncoorr_V_SMARTS, color='green', linestyle='--', linewidth=3, alpha=0.5)
axes[0].axhline(y=uncoorr_V_LCO, color='black', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_V_SMARTS, color='green', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_V_LCO, color='black', linestyle='--', linewidth=3, alpha=0.5)

plt.tight_layout()
plt.legend()
plt.show()


#%%


#I band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

    # ---- LEFT: original ----
axes[0].errorbar(
    final_I['nice time'],
    final_I['Rmag'],
    yerr=final_I['e_Rmag'],
    fmt='.',
    color='chocolate',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    ip_LCO['nice time'],
    ip_LCO['mag'],
    yerr=ip_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO ip'
)


axes[0].set_ylabel("I mag")
axes[1].set_ylabel("I mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    I_new['nice time'],
    I_new['Rmag_corr'],
    yerr=I_new['e_Rmag'],
    fmt='.',
    color='chocolate',
    alpha=0.8,
    label='SMARTS'#v band

)

#LCO stuff
axes[1].errorbar(
    ip_LCO_corr['nice time'],
    ip_LCO_corr['mag_corr'],
    yerr=ip_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO ip'
)


# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
#ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax+8, ymin)
ax_right.set_ylim(ymax+8, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

axes[0].axhline(y=uncoorr_I_SMARTS, color='chocolate', linestyle='--', linewidth=3, alpha=0.5)
axes[0].axhline(y=uncoorr_ip_LCO, color='black', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_I_SMARTS, color='chocolate', linestyle='--', linewidth=3, alpha=0.5)
axes[1].axhline(y=coorr_ip_LCO, color='black', linestyle='--', linewidth=3, alpha=0.5)

plt.tight_layout()
plt.legend()
plt.show()



#%%

#correct the LCO band
offsets = {
    "R_LCO": coorr_R_SMARTS - coorr_R_LCO,
    "Rp_LCO": coorr_R_SMARTS - coorr_Rp_LCO,
    "V_SMARTS": coorr_V_LCO - coorr_V_SMARTS,
    "ip_LCO": coorr_I_SMARTS - coorr_ip_LCO,
}

R_LCO_corr["mag_shifted"] = R_LCO_corr["mag_corr"] + offsets['R_LCO']
Rp_LCO_corr["mag_shifted"] = Rp_LCO_corr["mag_corr"] + offsets['Rp_LCO']
V_new["Rmag_shifted"] = V_new["Rmag_corr"] + offsets['V_SMARTS']
ip_LCO_corr["mag_shifted"] = ip_LCO_corr["mag_corr"] + offsets['ip_LCO']

R_LCO_corr["alt_mag_shifted"] = R_LCO_corr["alt_mag_corr"] + offsets['R_LCO']
Rp_LCO_corr["alt_mag_shifted"] = Rp_LCO_corr["alt_mag_corr"] + offsets['Rp_LCO']
V_new["alt_Rmag_shifted"] = V_new["Rmag Divided Version"] + offsets['V_SMARTS']
ip_LCO_corr["alt_mag_shifted"] = ip_LCO_corr["alt_mag_corr"] + offsets['ip_LCO']


#R band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

R_new = corrected.get("R")
    # ---- LEFT: original ----
axes[0].errorbar(
    R_new['nice time'],
    R_new['Rmag_corr'],
    yerr=R_new['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    R_LCO_corr['nice time'],
    R_LCO_corr['mag_corr'],
    yerr=R_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[0].errorbar(
    Rp_LCO_corr['nice time'],
    Rp_LCO_corr['mag_corr'],
    yerr=Rp_LCO_corr['mag_corr_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)


axes[0].set_ylabel("R mag")
axes[1].set_ylabel("R mag")


# ---- RIGHT: subtracted ----
axes[1].errorbar(
    R_new['nice time'],
    R_new['Rmag_corr'],
    yerr=R_new['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    R_LCO_corr['nice time'],
    R_LCO_corr['mag_shifted'],
    yerr=R_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[1].errorbar(
    Rp_LCO_corr['nice time'],
    Rp_LCO_corr['mag_shifted'],
    yerr=Rp_LCO_corr['mag_corr_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)

# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()


#R band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')
    # ---- LEFT: original ----
axes[0].errorbar(
    V_new['nice time'],
    V_new['Rmag_corr'],
    yerr=V_new['e_Rmag'],
    fmt='.',
    color='green',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    v_LCO_corr['nice time'],
    v_LCO_corr['mag_corr'],
    yerr=v_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO V'
)


axes[0].set_ylabel("V mag")
axes[1].set_ylabel("V mag")


# ---- RIGHT: subtracted ----
axes[1].errorbar(
    V_new['nice time'],
    V_new['Rmag_shifted'],
    yerr=V_new['e_Rmag'],
    fmt='.',
    color='green',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    v_LCO_corr['nice time'],
    v_LCO_corr['mag_corr'],
    yerr=v_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO V'
)


# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()



#R band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')
    # ---- LEFT: original ----
axes[0].errorbar(
    I_new['nice time'],
    I_new['Rmag_corr'],
    yerr=I_new['e_Rmag'],
    fmt='.',
    color='chocolate',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    ip_LCO_corr['nice time'],
    ip_LCO_corr['mag_corr'],
    yerr=ip_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO ip'
)


axes[0].set_ylabel("I mag")
axes[1].set_ylabel("I mag")


# ---- RIGHT: subtracted ----
axes[1].errorbar(
    I_new['nice time'],
    I_new['Rmag_corr'],
    yerr=I_new['e_Rmag'],
    fmt='.',
    color='chocolate',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    ip_LCO_corr['nice time'],
    ip_LCO_corr['mag_shifted'],
    yerr=ip_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO ip'
)


# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()

#%%
#save files


'''

band='R'


header = f"# MJD corrected {band} MAG corrected uncertainty upperlimitflag {band} MAG uncertainty alt corrected MAG"
for ftype in ['banzai']:#['orac','banzai']:
    tosave=Rp_LCO_corr#.loc[ip_LCO_corr['from']==ftype]
    print(len(tosave))
    tosave=tosave[['MJD', 'mag_shifted', 'mag_corr_err', 'flag', 'mag', 'mag_err','alt_mag_shifted']]
    with open(f'/home/kmc249/Downloads/subtracted_shifted_LCO/rp_usable_{ftype}_corrected.txt', "w") as file:
        file.write(header + "\n")
        
        tosave.to_csv(
            file,
            sep=" ",
            index=False,
            header=False,
        )
'''