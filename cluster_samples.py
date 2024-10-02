import os
import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='snli')
parser.add_argument('--cluster_type', type=str, default='manual')
parser.add_argument('--target_split', type=str, default='train')
parser.add_argument('--model', type=str, default='')

args = parser.parse_args()


if 'roberta' in args.model:

    if args.dataset == 'snli':
        df = pd.read_csv('results/test/nli-v4:snli-roberta-base-train-train_stats.csv')

    elif args.dataset == 'anli':
        df = pd.read_csv('results/test/nli-v3:anli-roberta-base-test-test_stats.csv')

    elif args.dataset == 'multi_nli':
        df = pd.read_csv('results/test/nli-v4:multi_nli-roberta-base-train-train_stats.csv')

    elif 'fever' in args.dataset:
        df = pd.read_csv('results/test/nli-v4:fever-roberta-base-dev-stats.csv')
    else:
        raise ValueError(f'Unknown dataset = {args.dataset}')

elif 'deberta' in args.model:
    if args.dataset == 'snli':
        df = pd.read_csv('results/test/nli-v5:snli-deberta-test-stats.csv')

    elif args.dataset == 'multi_nli':
        df = pd.read_csv('results/test/nli-v5:multi_nli-deberta-validation-stats.csv')

    elif 'fever' in args.dataset:
        df = pd.read_csv('results/test/nli-v5:fever-deberta-dev-stats.csv')
    else:
        raise ValueError(f'Unknown dataset = {args.dataset}')


df['is-correct-roberta'] = df[f'{args.model}-prediction'] == df['true_label']
df['is-correct-roberta-hypo'] = df[f'{args.model}-hypo-prediction'] == df['true_label']
df['true_label_name'] = df['true_label'].replace(2, 'contradiction').replace(1, 'neutral').replace(0, 'entailment')

heuristic_names = ['heuristic:contains_negation', 'heuristic:misspelled_words', 'heuristic:number_of_antonyms', 'heuristic:word_overlap', 'heuristic:length_missmatch']

def get_performance(df, key):
    y_true = df['true_label'].values
    y_pred = df[key].values
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
    }

def plot_heuristics_per_cluster(df, ax, title):
    for i, heuristic_name in enumerate(heuristic_names):
        sns.boxplot(x='hardness', y=heuristic_name, data=df, ax=ax[i], order = ['easy', 'ambigous', 'hard'], showfliers=False)
        ax[i].set_xlabel('Hardness')
        ax[i].set_ylabel('Heuristic Value')
        ax[i].set_title(heuristic_name.replace('heuristic:', '').replace('_', ' ').title())

def plot_feature_values_per_cluster(df, features, ax, title):
    ax[0].set_ylabel('Feature Value')

    for i, feature_name in enumerate(features):
        sns.boxplot(x='hardness', y=feature_name, data=df, ax=ax[i], order = ['easy', 'ambigous', 'hard'], showfliers=False)
        ax[i].set_xlabel('Hardness')
        ax[i].set_title(feature_name.replace('heuristic:', '').replace('_', ' ').title())

def cluster_samples(df, features, order_by, cluster_type = 'kmeans'):
    print("::: Clustering samples ...", cluster_type, features, order_by)
    if order_by not in features:
        raise ValueError(f'order_by = {order_by} must be in features = {features}')

    if cluster_type == 'kmeans':
        kmeans = KMeans(n_clusters=3, n_init = 10, verbose = 2).fit_predict(StandardScaler().fit_transform(df[features]))
        clusters = kmeans

    elif cluster_type == 'gaussian_mixture':
        kmeans = GaussianMixture(n_components=3, n_init = 10, verbose = 2).fit_predict(StandardScaler().fit_transform(df[features]))
        clusters = kmeans

    elif cluster_type == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward').fit(StandardScaler().fit_transform(df[features]))
        clusters = clustering.labels_

    elif cluster_type == 'manual':
        clusters = np.zeros(df.shape[0])
        clusters[
            (df['both_mean_conf'] <= np.percentile(df['both_mean_conf'], 0.33)) &
            (df['both_std_conf'] <= np.percentile(df['both_std_conf'], 0.33))
        ] = 1 # hard

        clusters[
            (df['both_mean_conf'] >= np.percentile(df['both_mean_conf'], 0.66)) &
            (df['both_std_conf'] <= np.percentile(df['both_std_conf'], 0.33))
        ] = 2 # easy
    else:
        raise ValueError(f'Unknown cluster_type = {cluster_type}')

    df['hardness'] = clusters
    average_conf_per_cluster = df.groupby('hardness')[order_by].mean()

    easy_cluster = average_conf_per_cluster.idxmax(axis=0)
    hard_cluster = average_conf_per_cluster.idxmin(axis=0)

    clusters = set([0, 1, 2])
    ambigous_cluster = clusters.difference(set([easy_cluster, hard_cluster])).pop()
    df = df.copy()
    df['hardness'] = df['hardness'].replace(easy_cluster, 'easy')
    df['hardness'] = df['hardness'].replace(hard_cluster, 'hard')
    df['hardness'] = df['hardness'].replace(ambigous_cluster, 'ambigous')

    return df

features_all_both = ['both_mean_conf', 'both_std_conf', 'both_average_margin', 'both_average_correctness']
features_all_hypo = ['hypo_mean_conf', 'hypo_std_conf', 'hypo_average_margin', 'hypo_average_correctness']

features_all = features_all_both + features_all_hypo

df_all = cluster_samples(df.copy(), features = features_all, order_by = 'both_mean_conf', cluster_type = args.cluster_type)

# Performance per split
print('Performance per split')
print("all", get_performance(df_all, key = f'{args.model}-prediction'))
print('easy', get_performance(df_all[df_all['hardness'] == 'easy'], key = f'{args.model}-prediction'))
print('ambiguous', get_performance(df_all[df_all['hardness'] == 'ambigous'], key = f'{args.model}-prediction'))
print('hard', get_performance(df_all[df_all['hardness'] == 'hard'], key = f'{args.model}-prediction'))

print("Performance Hypo per Split")
print("all", get_performance(df_all, key = f'{args.model}-hypo-prediction'))
print('easy', get_performance(df_all[df_all['hardness'] == 'easy'], key = f'{args.model}-hypo-prediction'))
print('ambiguous', get_performance(df_all[df_all['hardness'] == 'ambigous'], key = f'{args.model}-hypo-prediction'))
print('hard', get_performance(df_all[df_all['hardness'] == 'hard'], key = f'{args.model}-hypo-prediction'))

print("Percent easy:", df_all[df_all['hardness'] == 'easy'].shape[0] / df_all.shape[0])
print("Percent ambiguous:", df_all[df_all['hardness'] == 'ambigous'].shape[0] / df_all.shape[0])
print("Percent hard:", df_all[df_all['hardness'] == 'hard'].shape[0] / df_all.shape[0])


# Save the clustering (overkill but whatever)
df_final = df.copy()
df = cluster_samples(df.copy(), features = features_all, order_by = 'both_mean_conf', cluster_type = 'manual')
df_final['hardness:manual'] = df['hardness']

df = cluster_samples(df.copy(), features = features_all, order_by = 'both_mean_conf', cluster_type = 'kmeans')
df_final['hardness:kmeans'] = df['hardness']

df = cluster_samples(df.copy(), features = features_all, order_by = 'both_mean_conf', cluster_type = 'hierarchical')
df_final['hardness:hierarchical'] = df['hardness']

df = cluster_samples(df.copy(), features = features_all, order_by = 'both_mean_conf', cluster_type = 'gaussian_mixture')
df_final['hardness:gaussian_mixture'] = df['hardness']

df_final.to_csv(f'assets/{args.model}-{args.dataset}-{args.target_split}_final.csv', sep=',', index=False)
