#!/bin/bash
set -e
cd ..

GROUP=our_nli-baselines
WANDB_MODE=run

python main.py --hardness easy --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-easy --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness easy ambigous --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-easy-amb --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness ambigous --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-amb --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness ambigous hard --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-amb-hard --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness hard --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-hard --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness easy hard --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-easy-hard --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128

python main.py --hardness easy --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-easy --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness easy ambigous --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-easy-amb --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness ambigous --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-amb --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness ambigous hard --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-amb-hard --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness hard --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-hard --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --hardness easy hard --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-easy-hard --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128

python main.py --config_file configs/our_multinli.yaml --do_sample 1 --sample_percent 0.45 --model_name roberta-base --name our_multi_nli-roberta-base-sample-1 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/our_multinli.yaml --do_sample 1 --sample_percent 0.45 --model_name roberta-base --name our_multi_nli-roberta-base-sample-2 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/our_multinli.yaml --do_sample 1 --sample_percent 0.45 --model_name roberta-base --name our_multi_nli-roberta-base-sample-3 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/our_snli.yaml --do_sample 1 --sample_percent 0.45 --model_name roberta-base --name our_snli-roberta-base-sample-1 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/our_snli.yaml --do_sample 1 --sample_percent 0.45 --model_name roberta-base --name our_snli-roberta-base-sample-2 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/our_snli.yaml --do_sample 1 --sample_percent 0.45 --model_name roberta-base --name our_snli-roberta-base-sample-3 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128

python main.py --config_file configs/our_multinli.yaml --do_sample 1 --sample_percent 0.33 --model_name roberta-base --name our_multi_nli-roberta-base-sample-33 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
python main.py --config_file configs/our_snli.yaml --do_sample 1 --sample_percent 0.33 --model_name roberta-base --name our_snli-roberta-base-sample-33 --group $GROUP --mode $WANDB_MODE --epochs 10 --batch_size 64 --use_amp 1 --output_dir test --eval_every_step 128
