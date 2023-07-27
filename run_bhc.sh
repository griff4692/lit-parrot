#!/usr/bin/bash

set -e

DATA_DIR="/nlp/data/cdr/epic_docs_2020_20230625/llama_data"
CKPT_DIR="checkpoints/meta-llama/Llama-2-7b-hf"
OUT_DIR="out/adapter_v2/bhc_llama"

python finetune/bhc_adapter_v2.py --data_dir $DATA_DIR --checkpoint_dir $CKPT_DIR --out_dir $OUT_DIR \
  --precision bf16-true
