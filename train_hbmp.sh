#!/bin/bash

python3 train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus scitail \
  --encoder_type HBMP \
  --activation leakyrelu \
  --optimizer adam \
  --word_embedding glove.840B.300d \
  --embed_dim 300 \
  --fc_dim 600 \
  --hidden_dim 600 \
  --layers 1 \
  --dropout 0.1 \
  --learning_rate 0.0005 \
  --lr_patience 1 \
  --lr_decay 0.99 \
  --lr_reduction_factor 0.2 \
  --weight_decay 0 \
  --early_stopping_patience 3 \
  --save_path results \
  --seed 1234
