#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:56:54 2026

@author: kmc249
"""

import numpy as np
import pandas as pd
import glob
import os
import shutil
folder1 = "/scratch/temp_CD_data/AqlX-1"

#need to move 2011 data
data2011=glob.glob('/OLD-NET-DRIVE/smarts/buxton/AqlX1/optical/2011/*')

#need to move 2002 data
loc2002='/OLD-NET-DRIVE/xrb-archive/data/AqlX1-2006/processed_R'

def collect_filenames(root_dir):
    """Return a set of filenames (no paths) found recursively."""
    names = set()
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            names.add(f)
    return names

def filename_to_paths(root_dir):
    """
    Return a dict mapping filename -> list of full paths
    """
    mapping = {}
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            full_path = os.path.join(root, f)
            mapping.setdefault(f, []).append(full_path)
    return mapping


# Collect filenames
folder1_files = collect_filenames(folder1)
folder2_files = collect_filenames(loc2002)
folder2_map = filename_to_paths(loc2002)


# Files in folder2 that are not anywhere in folder1
missing_files = folder2_files - folder1_files

# Report
print(f"Number of files in folder2 not found in folder1: {len(missing_files)}")
for f in sorted(missing_files):
    print(f)

dest_1m = "/scratch/temp_CD_data/AqlX-1/1m/rccd"
dest_13m = "/scratch/temp_CD_data/AqlX-1/1.3m/rccd"

for fname in missing_files:
    if fname.endswith(".fits") and not fname.startswith("rccd03"):
        # In case the same filename appears multiple times
        for src in folder2_map.get(fname, []):
            dst = os.path.join(dest_1m, fname)
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)
    elif fname.endswith(".fits") and fname.startswith("rccd03"):
        for src in folder2_map.get(fname, []):
            dst = os.path.join(dest_13m, fname)
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)


for src in data2011:
    if os.path.isfile(src):
        fname = os.path.basename(src)
        dst = os.path.join(dest_13m, fname)
        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)
