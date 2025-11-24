#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:15:44 2025

@author: kmc249
"""

import pandas as pd
import numpy as np

df=pd.read_csv('/home/kmc249/Downloads/aql_x1_outbursts.csv')

# Suppose df is your DataFrame
# Replace string "NAN" with np.nan
df.replace("NAN", np.nan, inplace=True)
exclude_cols = ['Start (MJD)', 'End (MJD)', 'First HM (MJD)', 'Second HM (MJD)']
plot_cols = [c for c in df.columns if c not in exclude_cols and c != df.columns[0]]  # exclude first column for plotting


# Convert all columns to numeric where possible
for col in df.columns[1:]:  # skip first column (labels)
    df[col] = pd.to_numeric(df[col], errors='coerce')

import matplotlib.pyplot as plt
import itertools

labels = df.iloc[:, 0]  # first column as labels
plt.figure(figsize=(20, 15))
n = len(plot_cols)
plot_idx = 1

for i, j in itertools.product(range(n), repeat=2):
    plt.subplot(n, n, plot_idx)
    
    x = df[plot_cols[j]]
    y = df[plot_cols[i]]
    
    # Mask to ignore NaNs
    mask = ~x.isna() & ~y.isna()
    plt.scatter(x[mask], y[mask])
    
    # Add labels to each point
    for k, txt in enumerate(labels[mask]):
        plt.text(x[mask].iloc[k], y[mask].iloc[k], str(txt), fontsize=8)
    
    if i == n-1:
        plt.xlabel(plot_cols[j])
    if j == 0:
        plt.ylabel(plot_cols[i])
    
    plot_idx += 1

plt.tight_layout()
plt.savefig('/home/kmc249/big_outburst_plot.png', dpi=250)
plt.show()
