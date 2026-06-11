"""Standalone environment sanity check (no dataset / no checkpoint needed).

Validates: registry import, config import, model build on CUDA, and a forward pass.
Run from repo root:  python scripts/env_check.py
"""
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import ever as er
from ever.core.config import import_config
from ever.core.builder import make_model

print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")

er.registry.register_all()
print("register_all: OK")

cfg = import_config("baseline.MetaRS")
p = cfg["model"]["params"]
print("config: encoder.num_classes=%s num_classes=%s begin_mmr_iter=%s loss=%s wd=%s" % (
    p["encoder"]["num_classes"], p["num_classes"], p["begin_mmr_iter"],
    list(p["loss"].keys()), cfg["optimizer"]["params"]["weight_decay"]))

# build without downloading ImageNet weights (architecture/CUDA check)
cfg["model"]["params"]["encoder"]["pretrained"] = False
model = make_model(cfg["model"]).cuda().eval()
n = sum(t.numel() for t in model.parameters())
print("model built on CUDA | #params %.3f M (author log 113.912 M)" % (n / 1e6))

# forward on a dummy SAR(1)+RGB(3) 512x512 batch
x = torch.randn(1, 4, 512, 512, device="cuda")
with torch.no_grad():
    y = model(x)
print("forward OK | out", tuple(y.shape), "| sums~1:", float(y[0, :, 0, 0].sum()))
assert y.shape == (1, 8, 512, 512), y.shape

# confirm ImageNet-pretrained resnet50 download path works (used by real training)
try:
    cfg2 = import_config("baseline.MetaRS")
    cfg2["model"]["params"]["encoder"]["pretrained"] = True
    _ = make_model(cfg2["model"])
    print("pretrained=True build: OK (resnet50 weights reachable)")
except Exception as e:
    print("pretrained=True build: FAILED ->", repr(e)[:200])

print("ENV_CHECK_OK")
