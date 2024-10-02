#!/bin/bash
set -e
cd ..

WANDB_MODE=run
GROUP=nli-v5

######################### SNLI ########################
# roberta-base
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 1 --use_hypo 0 --target_split train --name snli-roberta-base-train-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 1 --use_hypo 1 --target_split train --name snli-roberta-base-train-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
#########################################################

######################### MultiNLI ########################
# roberta-base
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 1 --use_hypo 0 --target_split train --name multi_nli-roberta-base-train-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 1 --use_hypo 1 --target_split train --name multi_nli-roberta-base-train-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
#########################################################


######################## NLI_FEVER ########################
python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name roberta-base --save_characterization 0 --use_hypo 0 --target_split train --name fever-roberta-base-train-both2 --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name roberta-base --save_characterization 0 --use_hypo 1 --target_split train --name fever-roberta-base-train-hypo2 --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1

python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name roberta-base --save_characterization 1 --use_hypo 0 --target_split dev --name fever-roberta-base-dev-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name roberta-base --save_characterization 1 --use_hypo 1 --target_split dev --name fever-roberta-base-dev-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
################################################

#####################################################################################################
#####################################################################################################
#####################################################################################################

######################## NLI_FEVER ########################
python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 0 --target_split dev --name fever-deberta-dev-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 1 --target_split dev --name fever-deberta-dev-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1

python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 0 --target_split train --name fever-deberta-train-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 1 --target_split train --name fever-deberta-train-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
################################################

# DeBERTa
######################### SNLI ########################
# deberta-base
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 0 --target_split test --name snli-deberta-test-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 1 --target_split test --name snli-deberta-test-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1

python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 0 --target_split train --name snli-deberta-train-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name snli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 1 --target_split train --name snli-deberta-train-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
#########################################################

######################### MultiNLI ########################
# deberta-base
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 0 --target_split validation --name multi_nli-deberta-validation-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 1 --target_split validation --name multi_nli-deberta-validation-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1

python main.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 0 --target_split train --name multi_nli-roberta-train-both --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
python main.py --/config_file configs/base_config.yaml --dataset_name multi_nli --model_name microsoft/deberta-v3-base --save_characterization 1 --use_hypo 1 --target_split train --name multi_nli-roberta-train-hypo --group $GROUP --mode $WANDB_MODE --epochs 5 --batch_size 32 --use_amp 1 --output_dir test --eval_every_step -1
#########################################################

