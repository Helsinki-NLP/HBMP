#!/bin/bash

python3 train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus snli \
  --encoder_type BiLSTMMaxPoolEncoder \
  --activation tanh \
  --optimizer sgd \
  --word_embedding glove.840B.300d \
  --embed_dim 300 \
  --fc_dim 512 \
  --hidden_dim 2048 \
  --layers 1 \
  --dropout 0 \
  --learning_rate 0.1 \
  --lr_patience 1 \
  --lr_decay 0.99 \
  --lr_reduction_factor 0.2 \
  --save_path results \
  --seed 1234
