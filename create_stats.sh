#!/bin/bash
set -e

python cluster_samples.py --dataset snli --cluster_type gaussian_mixture --model roberta-base
python cluster_samples.py --dataset multi_nli --cluster_type gaussian_mixture --model roberta-base
python cluster_samples.py --dataset fever --cluster_type gaussian_mixture --model roberta-base

python cluster_samples.py --dataset snli --cluster_type gaussian_mixture --model deberta
python cluster_samples.py --dataset multi_nli --cluster_type gaussian_mixture --model deberta
python cluster_samples.py --dataset fever --cluster_type gaussian_mixture --model deberta
