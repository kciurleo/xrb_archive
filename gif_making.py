#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 10:14:00 2026

@author: kmc249
"""

from pathlib import Path
from PIL import Image
from datetime import datetime

folder = Path("/home/kmc249/Downloads/aqlpics")

def make_gif(prefix, output_path, duration_ms=200):
    files = []

    for f in folder.glob(f"{prefix}_*.png"):
        ts_str = f.stem.split("_", 1)[1]
        ts = datetime.fromisoformat(ts_str)
        files.append((ts, f))

    files.sort(key=lambda x: x[0])

    images = [Image.open(f).convert("RGBA") for _, f in files]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )

    print("Saved:", output_path)

make_gif("ap", folder / "ap.gif")
make_gif("sub", folder / "sub.gif")

