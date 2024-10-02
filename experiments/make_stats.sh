#!/bin/bash
set -e
cd ..

GROUP=nli-v4


# Only roberta-base on snli test set
# ######################### SNLI ########################
# # roberta-base
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name snli --model_name microsoft/deberta-v3-base --save_characterization 1 --target_split test --name snli-deberta-test --group $GROUP --use_amp 1 --output_dir test
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 1 --target_split train --name snli-roberta-base-train --group $GROUP --use_amp 1 --output_dir test
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name snli --model_name roberta-base --save_characterization 1 --target_split test --name snli-roberta-base-test --group $GROUP --output_dir test
# #########################################################


# ######################## MultiNLI ########################
# # roberta-base
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 1 --target_split validation --name multi_nli-roberta-base-val --group $GROUP --output_dir test
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name roberta-base --save_characterization 1 --target_split train --name multi_nli-roberta-base-train --group $GROUP --use_amp 1 --output_dir test
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name multi_nli --model_name microsoft/deberta-v3-base --save_characterization 1 --target_split validation --name multi_nli-deberta-validation --group $GROUP --use_amp 1 --output_dir test
# ################################################

######################### FEVER ########################
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name roberta-base --save_characterization 1 --target_split dev --name fever-roberta-base-dev --group $GROUP --use_amp 1 --output_dir test
python characterize_examples_clear_text.py --config_file configs/base_config.yaml --dataset_name pietrolesci/nli_fever --model_name microsoft/deberta-v3-base --save_characterization 1 --target_split dev --name fever-deberta-dev --group $GROUP --use_amp 1 --output_dir test