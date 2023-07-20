#!/usr/bin/bash

set -e

ORACLE_STRATEGY=$1
python finetune/bhc_adapter_v2.py --data_dir "data/bhc_${ORACLE_STRATEGY}" --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --out_dir "out/adapter_v2/bhc_alpaca_${ORACLE_STRATEGY}" --precision bf16-true
