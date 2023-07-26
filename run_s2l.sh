#!/usr/bin/bash

set -e

export NUM_DEVICES=$1
DATASET=$2
PARROT_MODEL=$3

if [[ $PARROT_MODEL == "falcon" ]]
then
  echo "Falcon"
  CKPT="tiiuae/falcon-7b"
  DATA="falcon"
elif [[ $PARROT_MODEL == "llama_chat" ]]
then
  echo "LLAMA Chat"
  DATA="llama"
  CKPT="meta-llama/Llama-2-7b-chat-hf"
else
  echo "LLAMA"
  DATA="llama"
  CKPT="meta-llama/Llama-2-7b-hf"
fi

python finetune/s2l_adapter_v2.py --data_dir "data/${DATASET}_${DATA}" --checkpoint_dir "checkpoints/${CKPT}" \
    --out_dir "out/adapter_v2/${DATASET}_${PARROT_MODEL}" --precision bf16-true
