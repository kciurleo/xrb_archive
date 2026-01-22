#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:59:29 2025

@author: kmc249
"""

import os
import numpy as np
import re
from collections import Counter
import zipfile
from datetime import datetime

#get directories
base_dirs = ['/USB4/archive/', '/USB3/archive/', '/USB2/archive/']

sub_dirs_4 = ['20170901ThruEnd', '20171001ThruEnd', '20171101ThruEnd', '20171201ThruEnd']
sub_dirs_3 = ['20170101ThruEnd', '20170201ThruEnd','20170301ThruEnd','20170401ThruEnd',
              '20170501ThruEnd','20170601ThruEnd','20170701ThruEnd','20170801ThruEnd','20170901ThruEnd']

all_dirs = [os.path.join('/USB4/archive/', d) for d in sub_dirs_4] + [os.path.join('/USB3/archive/', d) for d in sub_dirs_3]
           
# Patterns to match
filelist = []
pattern = re.compile(r'binir.*(dark|dome)', re.IGNORECASE)
seen_filenames = set()

for root_dir in all_dirs:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if pattern.search(filename) and filename not in seen_filenames:
                seen_filenames.add(filename)
                filelist.append(os.path.join(dirpath, filename))
                
print(len(filelist))

# Calculate total size in bytes
total_size_bytes = sum(os.path.getsize(f) for f in filelist)

# Convert to more readable units
total_size_kb = total_size_bytes / 1024
total_size_mb = total_size_kb / 1024
total_size_gb = total_size_mb / 1024

print(f"Total files: {len(filelist)}")
print(f"Total size: {total_size_bytes} bytes")
print(f"Total size: {total_size_kb:.2f} KB")
print(f"Total size: {total_size_mb:.2f} MB")
print(f"Total size: {total_size_gb:.2f} GB")

#count up files by year
year_counts = Counter()

for file in filelist:
    filename = file.split('/')[-1]
    match = re.search(r'(\d{2})\d{4}', filename)
    if match:
        year = match.group(1)
        year_counts[year] += 1
    else:
        year_counts["no_year"] += 1

print(year_counts)

#zip files
zip_output = "/home/kmc249/Downloads/cals_2017.zip"
skipped = []

with zipfile.ZipFile(
    zip_output,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    allowZip64=True
) as zipf:

    for path in set(filelist):
        print('writing ',path)
        if os.path.isfile(path):
            zipf.write(path, arcname=os.path.basename(path))
        else:
            skipped.append(path)

print(f"ZIP created: {zip_output}")
print(f"Files skipped: {len(skipped)}")
