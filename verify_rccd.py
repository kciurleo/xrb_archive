#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:47:38 2025

@author: kmc249
"""
import glob
import shutil
import os
import astropy.io.fits as fits
from lookup_name import *

workdir='/scratch/temp_CD_data'
folders=glob.glob(f'{workdir}/*')
print(folders)

def get_dest(tele, ir, rccd, ccd, hdr, r):
    #try to do some sleuthing via date if need be first
    tele = tele.strip().lower() if isinstance(tele, str) else 'none'
    if tele == 'none':
        yr = int(hdr['DATE-OBS'].split('-')[0])
        instrume = hdr.get('INSTRUME', '').lower()
        if 'andicam' in instrume and yr < 2003:
            tele = 'yalo 1m'
        elif 'andicam' in instrume and yr >= 2003:
            tele = 'ct13m'

    #otherwise find its destination
    if tele == 'ct13m' and ir:
        dest = '1.3m/ir'
    elif tele == 'ct13m' and (rccd or ccd):
        dest = '1.3m/rccd'
    elif tele in ['yalo 1m', 'yale 1m'] and (rccd or ccd or r):
        dest = '1m/rccd'
    elif tele in ['yalo 1m', 'yale 1m'] and ir:
        dest = '1m/ir'
    else:
        raise Exception(f"Telescope '{tele}' or opt/ir not recognized")

    return dest
#do the other guys first i guess
for folder in folders:
      if folder == f'{workdir}/other':
          #deal with other folder; we're just gonna toss the non-fits files
          files = glob.glob(f'{workdir}/other/**/*', recursive=True)
          files = [f for f in files if os.path.isfile(f)]
          for file in files:
              basef=file.split('/')[-1]
              if not (file.endswith('.fits') or file.endswith('.fits.gz')):
                  shutil.move(file, f'{workdir}/other/trash/{basef}')
                  print(f'trashed {basef} in other')
                  continue
              else:
                  hdr=fits.getheader(file)
                  obj=get_proper_name(hdr['OBJECT'])
                  if obj in folders:
                      shutil.move(file,f'{workdir}/{obj}/{basef}')
                      print(f'moved {basef} to {obj} lol good luck')
                  elif is_xrb_quick(obj):
                      os.makedirs(f'{workdir}/{obj}/1.3m', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/1.3m/rccd', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/1.3m/ir', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/1m/rccd', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/1m/ir', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/junk', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/bad_hdr', exist_ok=True)
                      os.makedirs(f'{workdir}/{obj}/trash', exist_ok=True)
                      shutil.move(file,f'{workdir}/{obj}/{basef}')
                      print(f'moved {basef} to {obj} lol good luck')
                  else:
                      shutil.move(file,f'{workdir}/other_nonxrb/{basef}')
                      print(f'moved {basef} to other_nonxrb')

for folder in folders:
    if folder == f'{workdir}/other':
        continue
    fname = folder.split('/')[-1]
    print(fname)

    os.makedirs(f'{folder}/1.3m', exist_ok=True)
    os.makedirs(f'{folder}/1.3m/rccd', exist_ok=True)
    os.makedirs(f'{folder}/1.3m/ir', exist_ok=True)
    os.makedirs(f'{folder}/1m/rccd', exist_ok=True)
    os.makedirs(f'{folder}/1m/ir', exist_ok=True)
    os.makedirs(f'{folder}/junk', exist_ok=True)
    os.makedirs(f'{folder}/bad_hdr', exist_ok=True)
    os.makedirs(f'{folder}/trash', exist_ok=True)

    files = glob.glob(f'{folder}/**/*', recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    for file in files:
        basef=file.split('/')[-1]
        #get rid of all non fits files
        if not (basef.endswith('.fits') or basef.endswith('.fits.gz')):
            shutil.move(file, f'{folder}/junk/{basef}')
            print(f'Moved {basef} to junk')
            continue
        #if fits file, first check to see if obj name matches
        try:
            hdr=fits.getheader(file)
        except:
            shutil.move(file, f'{folder}/bad_hdr/{basef}')
            print(f'Bad header. Moved {basef} to bad_hdr')
            continue
        try:
            hname=get_proper_name(hdr['OBJECT'])
        except:
            shutil.move(file, f'{workdir}/other_nonxrb/{basef}')
            print(f'Name not recognized. Moved {basef} to other_nonxrb')
            continue
        if hname!=fname:
            shutil.move(file, f'{workdir}/other/{basef}')
            print(f'Incorrect object. Moved {basef} to other')
            continue
        
        #get some useful info
        gz = basef.endswith('.fits.gz')
        tele=hdr['TELESCOP']
        ir=(basef.startswith('ir') or basef.startswith('binir'))
        rccd=basef.startswith('rccd')
        ccd=basef.startswith('ccd')
        r=basef.startswith('r')
        
        #destination based on the above
        try:
            dest=f'{folder}/{get_dest(tele, ir, rccd, ccd, hdr, r)}'
        except:
            #if something goes wrong, quarantine it to junk
            shutil.move(file, f'{folder}/junk/{basef}')
            print(f'Telescope {tele} or beginning not recognized. Moved {basef} to junk')
            continue
        
        #if it's a .gz file, we're going to see if the original is in place
        #same is true if ccd
        newbasef=basef
        if gz:
            newbasef=newbasef[:-3]
        if ccd:
            newbasef=f'r{newbasef}'
            
        #if the file or its base vers doesn't already exist, then put it where it belongs!
        fexist=(os.path.isfile(f'{dest}/{newbasef}') or os.path.isfile(f'{dest}/{basef}'))
        
        if not fexist and not ccd:
            shutil.move(file, f'{dest}/{basef}')
            print(f'Moved {basef} to {dest}')
            continue
        #make a ccd folder within the rccd folder only if need be
        elif not fexist and ccd:
            os.makedirs(f'{dest}/ccd', exist_ok=True)
            shutil.move(file, f'{dest}/ccd/{basef}')
            print(f'Moved {basef} to {dest}/ccd')
            continue
        #if the file already exists, move to trash!
        else:
            shutil.move(file, f'{folder}/trash/{basef}')
            print(f'Already have it. Moved {basef} to trash')
            continue
'''

# Set the path to your main directory
for main_dir in folders:
    print(main_dir)

    for root, dirs, files in os.walk(main_dir):
        # Skip the main directory itself
        if root == main_dir:
            continue
    
        for file in files:
            source = os.path.join(root, file)
            destination = os.path.join(main_dir, file)
    
            # If a file with the same name exists, rename to avoid overwrite
            if os.path.exists(destination):
                base, ext = os.path.splitext(file)
                i = 1
                while True:
                    new_name = f"{base}_{i}{ext}"
                    new_destination = os.path.join(main_dir, new_name)
                    if not os.path.exists(new_destination):
                        destination = new_destination
                        break
                    i += 1
    
            shutil.move(source, destination)
    
    print(f"All files moved to {main_dir}")
'''