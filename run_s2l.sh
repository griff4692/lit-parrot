#!/usr/bin/bash

set -e

export NUM_DEVICES=$1
DATASET=$2

python finetune/s2l_adapter_v2.py --data_dir "data/${DATASET}" --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --out_dir "out/adapter_v2/${DATASET}" --precision bf16-true
