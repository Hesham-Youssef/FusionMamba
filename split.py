

import os
import random
from pathlib import Path
import csv

def create_split_csv(src_dir, out_csv, val_ratio=0.2, seed=42):
    random.seed(seed)
    src_dir = Path(src_dir)
    scenes = sorted([p.name for p in src_dir.iterdir() if p.is_dir()])
    random.shuffle(scenes)

    n_val = int(len(scenes) * val_ratio)
    val_scenes = set(scenes[:n_val])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "split"])
        for scene in scenes:
            writer.writerow([scene, "val" if scene in val_scenes else "train"])

    print(f"Train scenes: {len(scenes) - n_val}, Val scenes: {n_val}")



create_split_csv("dataset/val", "splits.csv")