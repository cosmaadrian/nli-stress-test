#!/bin/bash
set -e
cd ..

GROUP=char_nli-baselines
WANDB_MODE=run

# Data Maps
python main.py --hardness_key hardness:datamaps --config_file configs/our_snli.yaml --model_name roberta-base --name datamaps_snli-roberta-base-easy --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness_key hardness:datamaps --config_file configs/our_multinli.yaml --model_name roberta-base --name datamaps_multi_nli-roberta-base-easy --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128

# AUM
python main.py --hardness_key hardness:aum --config_file configs/our_snli.yaml --model_name roberta-base --name aum_snli-roberta-base-easy --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness_key hardness:aum --config_file configs/our_multinli.yaml --model_name roberta-base --name aum_multi_nli-roberta-base-easy --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128