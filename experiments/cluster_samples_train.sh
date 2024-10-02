#!/bin/bash
set -e
cd ..

python cluster_samples.py --dataset snli --cluster_type gaussian_mixture --target_split train
python cluster_samples.py --dataset multi_nli --cluster_type gaussian_mixture --target_split train
python cluster_samples.py --dataset nli_fever --cluster_type gaussian_mixture --target_split train