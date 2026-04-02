#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:38:11 2026

@author: kmc249
"""

import os
import re
import shutil
from collections import Counter

#base_dirs = ['/USB4/archive/', '/USB3/archive/', '/USB2/archive/']
dest_root = '/neta/xrb/IRCALIBS'

#looking for binir darks and domes. maybe need to extend this to just ir?
#pattern = re.compile(r'binir.*(dark|dome)', re.IGNORECASE)
#pattern = re.compile(r'(binir|ir).*?(dark|dome|on|off)', re.IGNORECASE)

#for the tapes:
#base_dirs=['/OLD-NET-DRIVE/xrb-archive/data/', '/OLD-NET-DRIVE/xrb-archive/tapes/']#
base_dirs=['/scratch/']
pattern = re.compile(r'(binir|ir).*?(dark|dome|on|off|flat).*\.fits$', re.IGNORECASE)

year_pattern = re.compile(r'(\d{2})\d{4}')

filelist = []
seen_filenames = set()

#find files
for root_dir in base_dirs:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if pattern.search(filename) and filename not in seen_filenames:
                seen_filenames.add(filename)
                filelist.append(os.path.join(dirpath, filename))

print(f"Matched files: {len(filelist)}")

#assign years
year_counts = Counter()
skipped = []
collisions = []

for src_path in filelist:
    filename = os.path.basename(src_path)

    match = year_pattern.search(filename)
    if not match:
        skipped.append(src_path)
        year_counts["no_year"] += 1
        continue

    yy = int(match.group(1))

    if yy >= 90:
        year = f"19{yy}"
    else:
        year = f"20{yy:02d}"

    dest_dir = os.path.join(dest_root, year)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, filename)

    #don't overwrite existing files
    if os.path.exists(dest_path):
        collisions.append(dest_path)
        continue

    try:
        shutil.copy2(src_path, dest_path)
        year_counts[year] += 1
        print(f"moved: {src_path} → {dest_path}")
    except Exception as e:
        print(f"ERROR moving {src_path}: {e}")
        skipped.append(src_path)

print("\nSummary")
print("-------")
print(year_counts)
print(f"Skipped (no year or errors): {len(skipped)}")
print(f"Filename collisions: {len(collisions)}")