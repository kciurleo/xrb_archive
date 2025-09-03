#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 09:58:35 2025

@author: kmc249
"""

import os
import sys
import stat
from astropy.io import fits
import warnings
import ast
from lookup_name import *
import shutil

#ignore astropy warnings please
warnings.filterwarnings('ignore', category=UserWarning, append=True)

repo='/run/media/kmc249/'
dest='/scratch/nbn_temp/'


#allow for overwrites etc
for name in os.listdir(dest):
    full_path = os.path.join(dest, name)
    try:
        mode = os.stat(full_path).st_mode
        os.chmod(full_path, mode | stat.S_IWUSR)
    except Exception as e:
        print(f"Failed to modify {full_path}: {e}")

#rerun? for backups if a file sucks
try:
    rerun=sys.argv[1]
except:
    rerun=''

if rerun=='rerun':
    recopied=0
    refailed=0
    print('Rerunning.')
    refiles=ast.literal_eval(input("Input a list of specific files to copy over: "))
    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        for f in filenames:
            if f in refiles:
                full_path = os.path.join(dirpath, f)
                files.append(full_path)
    print(f'Copying files.')
    for file in files:
        basef=file.split('/')[-1]
        print(f'Copying {basef}')
        try:
            shutil.copy(file, dest+basef)
            recopied+=1
        except:
            print(f'Input/output error with {basef}')
            refailed+=1
    print(f'{recopied} files copied succesfully. {refailed} failed again.')

else:
    if rerun=='badhdrs':
        print('Bad header rerun')
        refiles=ast.literal_eval(input("Input a list of specific files with bad headers to check again: "))
        files = []
        for dirpath, dirnames, filenames in os.walk(repo):
            for f in filenames:
                if f in refiles:
                    full_path = os.path.join(dirpath, f)
                    files.append(full_path)
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(repo):
            for f in filenames:
                if f.endswith('.fits') or f.endswith('.fits.gz') :
                    full_path = os.path.join(dirpath, f)
                    files.append(full_path)
                
    targets= ast.literal_eval(input("Input a list of proper names of targets to pull: "))
                
    print('Reading headers...')
    bad_objects=[]
    skip_objects=[]
    copiednames=[]
    rereads=[]
    bad_hdrs=[]
    copied=0
    #read from the header into df
    for id, file in enumerate(files):
        print(f'Reading {file}')
        basef=file.split('/')[-1]
        #if reading the whole file is an issue and takes longer than 15 seconds
        try:
            hdr=fits.getheader(file)
        except:
            print(f'Could not read header for {basef}')
            bad_hdrs.append(basef)
            continue
        
        obj=hdr['OBJECT']
        #list out the bad objects
        try:
            propname=get_proper_name(obj)
        except:
            bad_objects.append(obj)
            continue
        #list out the not useful guys
        if propname in targets:
            print(f'Copying {basef}')
            try:
                shutil.copy(file, dest+basef)
                copied+=1
                copiednames.append(propname)
            except:
                print(f'Input/output error with {basef}')
                rereads.append(basef)
        else:
            skip_objects.append(propname)
            
    print(f'Copied {copied} files of objects including {set(copiednames)}.')
    print(f'The following objects were skipped: {set(skip_objects)}')
    print(f'The following objects failed to be recognized: {set(bad_objects)}')
    print(f'{len(set(rereads))} rereads needed: {rereads}')
    print(f'bad headers: {bad_hdrs}')

    