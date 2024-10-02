from lib.evaluator_extra import AcumenEvaluator
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
import torch
import os
from tqdm import tqdm
import numpy as np


class ClassificationOurStressEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(ClassificationOurStressEvaluator, self).__init__(args, model, logger = logger)

        from lib import nomenclature
        from lib import device

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[evaluator_args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, dataset_name = evaluator_args.dataset_name, kind = 'validation')
        self.device = device

    def trainer_evaluate(self, step = None):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, save = True):
        np.set_printoptions(suppress=True)
        y_pred = []
        y_true = []
        hardnesses = []
        length_missmatchs = []
        word_overlaps = []
        number_of_antonymss = []
        misspelled_wordss = []
        contains_negations = []
        idxs = []

        for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            output = self.model(batch)['nli']

            labels = batch['label'].detach().cpu().numpy().ravel().tolist()

            idxs.extend(batch['idx'].detach().cpu().numpy().ravel().tolist())
            hardnesses.extend(batch['hardness'])
            length_missmatchs.extend(batch['heuristic:length_missmatch'].detach().cpu().numpy().ravel().tolist())
            word_overlaps.extend(batch['heuristic:word_overlap'].detach().cpu().numpy().ravel().tolist())
            number_of_antonymss.extend(batch['heuristic:number_of_antonyms'].detach().cpu().numpy().ravel().tolist())
            misspelled_wordss.extend(batch['heuristic:misspelled_words'].detach().cpu().numpy().ravel().tolist())
            contains_negations.extend(batch['heuristic:contains_negation'].detach().cpu().numpy().ravel().tolist())
            y_true.extend(labels)
            y_pred.extend(output.labels.detach().cpu().numpy().ravel().tolist())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        idxs = np.array(idxs)

        results_df = pd.DataFrame({
            'idx': idxs,
            'y_pred': y_pred,
            'y_true': y_true,
            'hardness': hardnesses,
            'length_missmatch': length_missmatchs,
            'word_overlap': word_overlaps,
            'number_of_antonyms': number_of_antonymss,
            'misspelled_words': misspelled_wordss,
            'contains_negation': contains_negations,
        })

        # get results grouped by hardness
        final_results = []

        name = self.args.name
        group = self.args.group
        dataset = self.evaluator_args.dataset_name

        for hardness in results_df['hardness'].unique():
            df = results_df[results_df['hardness'] == hardness]

            final_results.append({
                'hardness': hardness,
                'accuracy': accuracy_score(df['y_true'], df['y_pred']),
                'precision': precision_score(df['y_true'], df['y_pred'], average = 'macro'),
                'recall': recall_score(df['y_true'], df['y_pred'], average = 'macro'),
                'f1': f1_score(df['y_true'], df['y_pred'], average = 'macro'),
                'zero_one_loss': zero_one_loss(df['y_true'], df['y_pred'], normalize = True),
                'name': name,
                'group': group,
                'dataset': dataset,
            })

        final_results_df = pd.DataFrame(final_results)

        if save:
            os.makedirs(f"results/{self.args.output_dir}", exist_ok = True)
            final_results_df.to_csv(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_our_{self.evaluator_args.dataset_name}_stress_results.csv", index = False)
            results_df.to_csv(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_our_{self.evaluator_args.dataset_name}_stress_results_full.csv", index = False)

        return final_results_df
