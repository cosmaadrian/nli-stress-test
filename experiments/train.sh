#!/bin/bash
set -e
cd ..

WANDB_MODE=dryrun
GROUP=nli-v5

######################### SNLI ########################
# roberta-base
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 1 --use_hypo 0 --target_split test --name snli-roberta-base-test-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 1 --use_hypo 1 --target_split test --name snli-roberta-base-test-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
#########################################################

######################## MultiNLI ########################
# roberta-base
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 1 --use_hypo 0 --target_split validation --name multi_nli-roberta-base-val-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 1 --use_hypo 1 --target_split validation --name multi_nli-roberta-base-val-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
################################################