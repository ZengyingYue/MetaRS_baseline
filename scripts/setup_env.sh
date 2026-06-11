#!/usr/bin/env bash
# One-shot environment setup for the MetaRS reproduction box.
# Verified on: Ubuntu 22.04, 2x RTX 4080 SUPER (sm_89), driver 580, python 3.10.
# Run from the repo root:  bash scripts/setup_env.sh
set -e

cd "$(dirname "$0")/.."

# 0) system venv support (root)
if ! python3 -c "import ensurepip" 2>/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv git
fi

# 1) venv
if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

# 2) torch (cu128 — matches the locally verified env)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3) the rest
pip install -r requirements_repro.txt

# 4) simplecv without its stale albumentations==0.4.2 pin
pip install --no-deps \
    "simplecv @ git+https://github.com/Z-Zheng/SimpleCV.git@4fa67581441ad150e82b3aa2c394a921f74e4ecd"

# 5) smoke check
python - <<'PY'
import torch, ever, simplecv, segmentation_models_pytorch, kmeans1d, albumentations, tifffile
import ever as er
er.registry.register_all()
print("env OK | torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| ever", getattr(ever, "__version__", "?"), "| albu", albumentations.__version__)
PY
echo "ENV_SETUP_DONE"
