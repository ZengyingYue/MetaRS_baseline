"""Pre-cut EarthMiss full images into 512x512 tiles for the author's eval protocol.

The author trained/evaluated MetaRS with pre-cut 512 tiles stored under
    <root>/<city>/images/SAR_512/
    <root>/<city>/images/RGB_512/
    <root>/<city>/masks_512/
(see author/config.pkl). The public EarthMiss release only ships the full
1024x1024 images under SAR/ RGB/ masks/, so this script regenerates the tiles.

By default it processes the three TEST cities used for evaluation and for the
MMR/ISW covariance source (America-NewYork, Japan-Hakodate, Peru-Callao).

Usage:
    EARTHMISS_ROOT=/path/to/EarthM3 python scripts/make_512_tiles.py
    python scripts/make_512_tiles.py --root /path/to/EarthM3 --cities America-NewYork ...
    python scripts/make_512_tiles.py --tile 512 --all   # tile every city

Tiles are non-overlapping on a `tile`-step grid; if a dimension is not divisible
by `tile`, the last row/column is aligned to the image edge (small overlap).
SAR/RGB keep their original (float32) dtype; masks keep uint8. Tile basenames are
identical across the three sub-folders so the dataloader's pairing asserts pass.
"""
import argparse
import glob
import os

import numpy as np
import tifffile
from skimage.io import imread

DEFAULT_TEST_CITIES = ["America-NewYork", "Japan-Hakodate", "Peru-Callao"]


def tile_starts(size, tile):
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, tile))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def cut(arr, top, left, tile):
    return arr[top:top + tile, left:left + tile]


def process_image_dir(src_dir, dst_dir, tile):
    """Returns dict basename(no-ext) -> list of (suffix, tile_array)."""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.tif")) +
                   glob.glob(os.path.join(src_dir, "*.png")))
    n_out = 0
    for fp in files:
        arr = imread(fp)
        h, w = arr.shape[0], arr.shape[1]
        base, ext = os.path.splitext(os.path.basename(fp))
        for ti, top in enumerate(tile_starts(h, tile)):
            for tj, left in enumerate(tile_starts(w, tile)):
                t = cut(arr, top, left, tile)
                out = os.path.join(dst_dir, f"{base}_{ti}_{tj}{ext}")
                tifffile.imwrite(out, np.ascontiguousarray(t))
                n_out += 1
    return len(files), n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("EARTHMISS_ROOT",
                    "/data/yhzhou23/data/EarthM3/"))
    ap.add_argument("--cities", nargs="*", default=None,
                    help="cities to process (default: 3 test cities)")
    ap.add_argument("--all", action="store_true",
                    help="process every city directory under root")
    ap.add_argument("--tile", type=int, default=512)
    args = ap.parse_args()

    if args.all:
        cities = sorted(d for d in os.listdir(args.root)
                        if os.path.isdir(os.path.join(args.root, d)))
    else:
        cities = args.cities or DEFAULT_TEST_CITIES

    print(f"root={args.root}  tile={args.tile}  cities={cities}")
    for city in cities:
        cdir = os.path.join(args.root, city)
        if not os.path.isdir(cdir):
            print(f"  [skip] missing city dir: {cdir}")
            continue
        jobs = [
            (os.path.join(cdir, "images", "SAR"), os.path.join(cdir, "images", "SAR_512")),
            (os.path.join(cdir, "images", "RGB"), os.path.join(cdir, "images", "RGB_512")),
            (os.path.join(cdir, "masks"),         os.path.join(cdir, "masks_512")),
        ]
        for src, dst in jobs:
            if not os.path.isdir(src):
                print(f"  [skip] missing src: {src}")
                continue
            n_in, n_out = process_image_dir(src, dst, args.tile)
            print(f"  {city}: {os.path.relpath(src, cdir)} -> "
                  f"{os.path.relpath(dst, cdir)}  ({n_in} imgs -> {n_out} tiles)")
    print("done.")


if __name__ == "__main__":
    main()
