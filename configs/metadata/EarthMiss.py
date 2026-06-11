import os
rgb_mean_v = [111.97261949846957, 118.42392668739762, 115.031314378774]
rgb_std_v = [72.97088063, 72.98526569, 81.91774359]
sar_mean_v = [63.30051921735858]
sar_std_v = [68.20405016]

# Author's original training path was "/data/yhzhou23/data/EarthM3/".
# Allow overriding via env var so the same code runs on the reproduction server
# without editing the config.
root_path = os.environ.get("EARTHMISS_ROOT", "/data/yhzhou23/data/EarthM3/")

# city list
train_cities = [
    "America-Eugene",
    "America-Louisville",
    "French-Paris",
    "Morocco-Casablanca",
    "Nanjing",
    "Netherlands-Rotterdam",
    "Singapore",
]

val_cities = [
    "Australia-PortHedland",
    "America-Pake",
    "Russian-Engels",
]

test_cities = [
    "America-NewYork",
    "Japan-Hakodate",
    "Peru-Callao",
]


def build_dirs(cities, mask_name="masks"):
    image_dirs = [os.path.join(root_path, city, "images") for city in cities]
    mask_dirs = [os.path.join(root_path, city, mask_name) for city in cities]
    return image_dirs, mask_dirs


train_image_dir, train_mask_dir = build_dirs(train_cities)

val_image_dir, val_mask_dir = build_dirs(val_cities)

test_image_dir, test_mask_dir = build_dirs(test_cities)

# Author's evaluation protocol uses pre-cut 512x512 tiles stored under
# `<city>/images/SAR_512`, `<city>/images/RGB_512` and `<city>/masks_512`
# (see author/config.pkl). The public EarthMiss release only ships the full
# 1024x1024 images, so these tiles must be generated with
# `scripts/make_512_tiles.py` before evaluation/MMR-covariance.
_, test_mask512_dir = build_dirs(test_cities, mask_name="masks_512")

