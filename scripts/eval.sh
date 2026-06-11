#!/usr/bin/env bash
# Faithful reproduction of the author's evaluation protocol:
#   - direct inference on the pre-cut SAR_512/RGB_512 test tiles (no sliding window)
#   - SAR-only branch (model returns softmax(sar_cls_pred))
# Generate the 512 tiles first:  python scripts/make_512_tiles.py
# Set EARTHMISS_ROOT to your dataset root if it is not the author's default path.

export EARTHMISS_ROOT=${EARTHMISS_ROOT:-/data/yhzhou23/data/EarthM3/}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

modality="SAR"
ckpt_path="./author/Best.pth"
config_path="baseline.MetaRS"

vis_path="vis-$(basename ${ckpt_path} .pth)"
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 1060 --use_env eval.py \
    --ckpt_path=${ckpt_path} \
    --config_path=${config_path} \
    --vis_path=${vis_path} \
    --modality=${modality}
    # add --slide for 1024 sliding-window inference instead of 512-tile direct
