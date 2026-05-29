#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:37:53 2026

@author: kmc249
"""

from pathlib import Path


target='J1753'
# Base directory
base = Path(f"/neta/xrb/{target}/1.3m/opt/rccd")

bands = ["B", "V", "R", "I"]

all_results = {}

for band in bands:
    raw_dir = base / band
    trimmed_dir = base / f"{band}_trimmed"

    # Original FITS files
    raw_files = sorted(raw_dir.glob("*.fits"))

    # Build expected trimmed filenames
    success = []
    failed = []

    for raw_file in raw_files:
        expected_trim = trimmed_dir / f"trim_{raw_file.name}"

        if expected_trim.exists():
            success.append(raw_file.name)
        else:
            failed.append(raw_file.name)

    all_results[band] = {
        "success": success,
        "failed": failed,
    }

    # Print summary
    print(f"\n=== {band} ===")
    print(f"Total raw files     : {len(raw_files)}")
    print(f"Successfully trimmed: {len(success)}")
    print(f"Failed to trim      : {len(failed)}")
    '''
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f}")
    '''