#!/bin/bash
set -e
cd ..


GROUP=our_nli-baselines

# 1. Evaluate models on Their StressTest
# 1.1 Evaluate models without using our cartography
GROUP=nli-baselines
STUFF="--config_file configs/base_config.yaml --model_name roberta-base --group $GROUP --batch_size 64 --output_dir results/"
python evaluate.py --eval_config configs/eval_theirs.yaml --name snli-roberta-base $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name multi_nli-roberta-base $STUFF

# # 1.2 Evaluate models using our cartography
GROUP=our_nli-baselines
STUFF="--config_file configs/our_multinli.yaml --model_name roberta-base --group $GROUP --batch_size 64 --output_dir results/"
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_multi_nli-roberta-base-hard $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_multi_nli-roberta-base-easy $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_multi_nli-roberta-base-easy-amb $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_multi_nli-roberta-base-amb $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_multi_nli-roberta-base-amb-hard $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_multi_nli-roberta-base-easy-hard $STUFF

# STUFF="--config_file configs/our_snli.yaml --model_name roberta-base --group $GROUP --batch_size 64 --output_dir results/"
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_snli-roberta-base-hard $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_snli-roberta-base-easy $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_snli-roberta-base-easy-amb $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_snli-roberta-base-amb $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_snli-roberta-base-amb-hard $STUFF
python evaluate.py --eval_config configs/eval_theirs.yaml --name our_snli-roberta-base-easy-hard $STUFF

# 2. Evaluate models on Our StressTest
# 2.1 Evaluate models without using our cartography
GROUP=nli-baselines
STUFF="--config_file configs/base_config.yaml --model_name roberta-base --group $GROUP --batch_size 64 --output_dir results/"
python evaluate.py --eval_config configs/eval_our.yaml --name snli-roberta-base $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name multi_nli-roberta-base $STUFF

# # 2.2 Evaluate models using our cartography
GROUP=our_nli-baselines
STUFF="--config_file configs/our_multinli.yaml --model_name roberta-base --group $GROUP --batch_size 64 --output_dir results/"
python evaluate.py --eval_config configs/eval_our.yaml --name our_multi_nli-roberta-base-hard $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_multi_nli-roberta-base-easy $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_multi_nli-roberta-base-easy-amb $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_multi_nli-roberta-base-amb $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_multi_nli-roberta-base-amb-hard $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_multi_nli-roberta-base-easy-hard $STUFF

STUFF="--config_file configs/our_snli.yaml --model_name roberta-base --group $GROUP --batch_size 64 --output_dir results/"
python evaluate.py --eval_config configs/eval_our.yaml --name our_snli-roberta-base-hard $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_snli-roberta-base-easy $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_snli-roberta-base-easy-amb $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_snli-roberta-base-amb $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_snli-roberta-base-amb-hard $STUFF
python evaluate.py --eval_config configs/eval_our.yaml --name our_snli-roberta-base-easy-hard $STUFF


GROUP=char_nli-baselines
# # Data Maps
python evaluate.py --config_file configs/our_snli.yaml --eval_config configs/eval_our.yaml --model_name roberta-base --name datamaps_snli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other
python evaluate.py --config_file configs/our_multinli.yaml --eval_config configs/eval_our.yaml --model_name roberta-base --name datamaps_multi_nli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other
# # AUM
python evaluate.py --config_file configs/our_snli.yaml --eval_config configs/eval_our.yaml --model_name roberta-base --name aum_snli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other
python evaluate.py --config_file configs/our_multinli.yaml --eval_config configs/eval_our.yaml --model_name roberta-base --name aum_multi_nli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other

# # Data Maps
python evaluate.py --config_file configs/our_snli.yaml --eval_config configs/eval_theirs.yaml --model_name roberta-base --name datamaps_snli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other
python evaluate.py --config_file configs/our_multinli.yaml --eval_config configs/eval_theirs.yaml --model_name roberta-base --name datamaps_multi_nli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other
# # AUM
python evaluate.py --config_file configs/our_snli.yaml --eval_config configs/eval_theirs.yaml --model_name roberta-base --name aum_snli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other
python evaluate.py --config_file configs/our_multinli.yaml --eval_config configs/eval_theirs.yaml --model_name roberta-base --name aum_multi_nli-roberta-base-easy --group $GROUP --batch_size 64 --output_dir results_other

GROUP=our_nli-baselines
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-1 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-2 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-3 --group $GROUP --batch_size 64 --output_dir results_sampled

python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-1 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-2 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-3 --group $GROUP --batch_size 64 --output_dir results_sampled

############################################################

python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-1 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-2 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-3 --group $GROUP --batch_size 64 --output_dir results_sampled

python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-1 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-2 --group $GROUP --batch_size 64 --output_dir results_sampled
python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-3 --group $GROUP --batch_size 64 --output_dir results_sampled

python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-33 --group $GROUP --batch_size 64 --output_dir results_sampled33
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_snli.yaml --model_name roberta-base --name our_snli-roberta-base-sample-33 --group $GROUP --batch_size 64 --output_dir results_sampled33
python evaluate.py --eval_config configs/eval_theirs.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-33 --group $GROUP --batch_size 64 --output_dir results_sampled33
python evaluate.py --eval_config configs/eval_our.yaml --config_file configs/our_multinli.yaml --model_name roberta-base --name our_multi_nli-roberta-base-sample-33 --group $GROUP --batch_size 64 --output_dir results_sampled33
