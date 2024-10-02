#!/bin/bash
set -e
cd ..

GROUP=nli-baselines
WANDB_MODE=run

######################## SNLI ########################
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 0 --name snli-roberta-base --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 256 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/base_config.yaml --use_hypo 1 --dataset_name snli --model_name roberta-base --save_characterization 0 --name snli-roberta-base-hypo --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 256 --use_amp 1 --output_dir test --eval_every_step 128

####################### MultiNLI ########################
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 0 --name multi_nli-roberta-base --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 256 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/base_config.yaml --use_hypo 1 --dataset_name multi_nli --model_name roberta-base --save_characterization 0 --name multi_nli-roberta-base-hypo --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 256 --use_amp 1 --output_dir test --eval_every_step 128