#!/bin/bash -l
#SBATCH -J SCITAIL_HBMP_2400D
#SBATCH -o out_HBMP_2400D_%J.txt
#SBATCH -e err_HBMP_2400D_%J.txt
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:p100:1
# run command
module purge
module load gcc cuda python-env/3.5.3-ml
module list

srun python3 train.py \
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

# This script will print some usage statistics to the
# end of file: output.txt
# Use that to improve your resource request estimate
# on later jobs.
used_slurm_resources.bash
