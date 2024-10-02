from lib.evaluator_extra import AcumenEvaluator
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
import torch
import os
from tqdm import tqdm
import numpy as np


class ClassificationStressEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(ClassificationStressEvaluator, self).__init__(args, model, logger = logger)

        from lib import nomenclature
        from lib import device

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[evaluator_args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, kind = 'validation')
        self.device = device

    def trainer_evaluate(self, step = None):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, save = True):
        np.set_printoptions(suppress=True)
        y_pred = []
        y_true_gold = []
        kinds = []
        idxs = []

        for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            output = self.model(batch)['nli']

            labels = batch['label'].detach().cpu().numpy().ravel().tolist()

            idxs.extend(batch['idx'].detach().cpu().numpy().ravel().tolist())
            kinds.extend(batch['kind'])
            y_true_gold.extend(labels)
            y_pred.extend(output.labels.detach().cpu().numpy().ravel().tolist())

        y_pred = np.array(y_pred)
        y_true_gold = np.array(y_true_gold)

        results_df = pd.DataFrame({
            'idx': idxs,
            'kind': kinds,
            'y_pred': y_pred,
            'y_true_gold': y_true_gold,
        })

        # get results grouped by kind
        final_results = []

        name = self.args.name
        group = self.args.group
        dataset = self.evaluator_args.dataset

        for kind in results_df['kind'].unique():
            kind_df = results_df[results_df['kind'] == kind]

            final_results.append({
                'kind': kind,
                'acc': accuracy_score(kind_df['y_true_gold'], kind_df['y_pred']),
                'prec': precision_score(kind_df['y_true_gold'], kind_df['y_pred'], average = 'macro'),
                'recall': recall_score(kind_df['y_true_gold'], kind_df['y_pred'], average = 'macro'),
                'f1': f1_score(kind_df['y_true_gold'], kind_df['y_pred'], average = 'macro'),
                'error': zero_one_loss(kind_df['y_true_gold'], kind_df['y_pred'], normalize = True),
                'name': name,
                'group': group,
                'dataset': dataset,
            })

        final_results_df = pd.DataFrame(final_results)

        if save:
            os.makedirs(f"results/{self.args.output_dir}", exist_ok = True)
            final_results_df.to_csv(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_stress_results.csv", index = False)
            results_df.to_csv(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_stress_results_full.csv", index = False)

        return final_results_df
