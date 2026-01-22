#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:17:32 2026

@author: kmc249
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import zipfile
from datetime import datetime

###testing to get object files for a given target
target='AqlX-1'
csv1=pd.read_csv(f'/home/kmc249/usbdrive_logs/usbdrivereplog_{target}.csv')
csv2=pd.read_csv(f'/home/kmc249/usbdrive_logs/usbdrivescrapelog_{target}.csv')
zip_output = "/home/kmc249/Downloads/aql_2017.zip"

#we need to deal with the replog separately, because we don't have the full file path. filename is the file name only
#alternatively: rerun scrape log and do it on All the things

#scrapelog:
#specifically get IR from 2017 because that's what I'll give Payaswini right now
interest=csv2.loc[(csv2['filename'].str.startswith(('binir'))) & (csv2['DATE-OBS'].str.startswith(('2017'))) ]
print(len(interest))
#get size 
def get_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0  

interest["size_bytes"] = interest["full path"].apply(get_size)

total_bytes = interest["size_bytes"].sum()

print(f"Total size: {total_bytes:,} bytes ({total_bytes / (1024**3):.2f} GB)")

'''
#zip files
skipped = []

with zipfile.ZipFile(
    zip_output,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    allowZip64=True
) as zipf:

    for path in set(interest["full path"]):
        print('writing ',path)
        if os.path.isfile(path):
            zipf.write(path, arcname=os.path.basename(path))
        else:
            skipped.append(path)

print(f"ZIP created: {zip_output}")
print(f"Files skipped: {len(skipped)}")
'''

#some code to help with replog from it which shall not be named:
usb_roots = ["/USB1/archive", "/USB2/archive", "/USB3/archive", "/USB4/archive"]

def parse_span_folder(folder_name):
    """
    Converts a folder like '20090101thru0131' to a start and end date
    as datetime.date objects. Assumes folder_name is 'YYYYMMDDthruMMDD'.
    """
    try:
        start_str, end_str = folder_name.split("thru")
        # Start date is full YYYYMMDD
        start_date = datetime.strptime(start_str, "%Y%m%d").date()
        # End date might be missing the year, assume same year as start
        if len(end_str) == 4:  # MMDD
            end_date = datetime.strptime(str(start_date.year) + end_str, "%Y%m%d").date()
        else:  # full YYYYMMDD
            end_date = datetime.strptime(end_str, "%Y%m%d").date()
        return start_date, end_date
    except:
        # Not a date range folder
        return None, None

def build_folder_map(usb_roots):
    """
    Recursively walk all archive folders and build a map of date ranges to folder paths.
    """
    folder_map = {}
    for root in usb_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for dirname in dirnames:
                if "thru" in dirname:
                    start, end = parse_span_folder(dirname)
                    if start and end:
                        folder_map[(start, end)] = os.path.join(dirpath, dirname)
    return folder_map

def find_file(file_date_str, filename, folder_map):
    """
    Given a file date like '20090105' and filename like 'binir090105.0296',
    find its full path using the folder_map.
    """
    file_date = datetime.strptime(file_date_str, "%Y%m%d").date()
    
    # Step 1: Find the span folder
    span_folder_path = None
    for (start, end), path in folder_map.items():
        if start <= file_date <= end:
            span_folder_path = path
            break
    
    if not span_folder_path:
        return None  # No folder found for this date
    
    # Step 2: Go into the daily folder named exactly like the date
    daily_folder = os.path.join(span_folder_path, file_date_str)
    if not os.path.isdir(daily_folder):
        return None  # Daily folder doesn't exist
    
    # Step 3: Check for the file in that folder
    file_path = os.path.join(daily_folder, filename)
    if os.path.isfile(file_path):
        return file_path
    else:
        return None

#Build the folder map once
folder_map = build_folder_map(usb_roots)
print(folder_map)

#Use it on all the guys:
for id, row in csv1.head(10).iterrows():
    file_date=row['replog'].split('/')[-1][6:-3]
    print(file_date)
    
    filename=row['Filename']+'.fits'
    print(filename)
    found_path = find_file(file_date, filename, folder_map)
    if found_path:
        print(f"File found: {found_path}")
    else:
        print("File not found.")

print(csv1.columns)
###the above doesn't actually work rn

