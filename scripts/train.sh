#!/usr/bin/env bash
# Author training recipe: 2 GPUs x batch_size 4 = effective batch 8, 15k iters,
# AdamW lr 1e-4 / wd 0.05, poly power 0.9. The overrides below reproduce the
# author's external params (see author/1776068983.6830268.log).
# Set EARTHMISS_ROOT to your dataset root if it is not the author's default path.
# Keep NUM_GPUS x batch_size == 8 to match the author's effective batch.
# Run scripts/make_512_tiles.py first (the MMR covariance at iter 1600 reads the
# SAR_512/RGB_512 test tiles).

export EARTHMISS_ROOT=${EARTHMISS_ROOT:-/data/yhzhou23/data/EarthM3/}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=2
export PYTHONPATH=$PYTHONPATH:`pwd`


config_path='baseline.MetaRS'
model_dir='./log/EarthMiss/MetaRS'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 1245 --use_env train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --trainer=th_ddp \
    train.eval_interval_epoch 20\
    train.save_ckpt_interval_epoch 20\
    data.train.params.batch_size  4 \
    data.train.params.CV.cur_k -1 \
    data.test.params.CV.cur_k -1 \
    train.num_iters 15000 \
    learning_rate.params.max_iters 15000  \
    learning_rate.params.base_lr 1e-4 \
    