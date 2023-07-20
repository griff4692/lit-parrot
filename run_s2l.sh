#!/usr/bin/bash

set -e

export NUM_DEVICES=$1
DATASET=$2
PARROT_MODEL=$3

if [[ $PARROT_MODEL == "falcon" ]]
then
  echo "Falcon"
  CKPT="tiiuae/falcon-7b"
else
  CKPT="meta-llama/Llama-2-7b-hf"
fi

python finetune/s2l_adapter_v2.py --data_dir "data/${DATASET}_${PARROT_MODEL}" --checkpoint_dir "checkpoints/${CKPT}" \
    --out_dir "out/adapter_v2/${DATASET}_${PARROT_MODEL}" --precision bf16-true
