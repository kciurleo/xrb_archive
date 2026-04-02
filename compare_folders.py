#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:05:25 2026

@author: kmc249
"""

import os
import sys
import glob

def get_all_filenames(base_folder):
    """
    Returns a set of filenames (ignoring directory structure)
    for all files inside base_folder.
    Treats .gz files as equivalent to their uncompressed version.
    """
    filenames = set()
    
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.gz'):
                file = file[:-3]  # remove '.gz'
            filenames.add(file)
    
    return filenames

def compare_folders(folder1, folder2):
    files1 = get_all_filenames(folder1)
    files2 = get_all_filenames(folder2)

    only_in_folder1 = files1 - files2

    if only_in_folder1:
        print(f"\nFilenames present in '{folder1}' but NOT in '{folder2}':\n")
        for file in sorted(only_in_folder1):
            print(file)
        return True
    else:
        print(f"\nNo unique filenames found in '{folder1}'. Everything exists in '{folder2}'.")
        return False

#
'''
hold_on=[]
delete_okay=[]
for folder in glob.glob('/scratch/temp_CD_data/*'):
    base=folder.split('/')[-1]
    if base=='AqlX-1':
        continue
    print(base)
    compare_folders(folder, f'/neta/xrb/{base}')
    yinp=input('continue?')
    #if compare_folders(folder, f'/neta/xrb/{base}'):
    #    hold_on.append(base)
    #else:
    #    delete_okay.append(base)
        

print(f'hold on: {hold_on}')
print(f'delete okay: {delete_okay}')
'''

compare_folders('/scratch/TAPE106/', '/neta/xrb/J0929-314/')