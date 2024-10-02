from lib.evaluator_extra import AcumenEvaluator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
import torch
import os
from tqdm import tqdm
import numpy as np
import json
from .logit_logger import GeneralLogger


class ClassificationEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(ClassificationEvaluator, self).__init__(args, model, logger = logger)

        from lib import nomenclature
        from lib import device

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[evaluator_args.dataset]

        if 'dataset_name' in evaluator_args:
            dataset_name = evaluator_args.dataset_name
        else:
            dataset_name = args.dataset_name

        if self.args.save_characterization:
            self.val_dataloader = self.dataset.val_dataloader(args, dataset_name = dataset_name, kind = args.target_split)
        else:
            self.val_dataloader = self.dataset.val_dataloader(args, dataset_name = dataset_name, kind = 'validation')

        self.device = device

        self.cartography = GeneralLogger()

    def trainer_evaluate(self, step = None):
        return self.evaluate(save = False, save_characterization = bool(self.args.save_characterization))

    @torch.no_grad()
    def evaluate(self, save = True, save_characterization = False):
        np.set_printoptions(suppress=True)
        y_pred = []
        y_true = []

        for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            output = self.model(batch)['nli']

            labels = batch['label'].detach().cpu().numpy().ravel().tolist()

            y_true.extend(labels)
            y_pred.extend(output.labels.detach().cpu().numpy().ravel().tolist())

            if save_characterization:
                if self.args.dataset_name == 'anli':
                    self.cartography.update(batch['uid'], output.logits.detach().cpu())

                elif self.args.dataset_name == 'snli':
                    self.cartography.update(batch['idx'].detach().cpu().numpy().tolist(), output.logits.detach().cpu())

                elif self.args.dataset_name == 'multi_nli':
                    self.cartography.update(batch['pairID'], output.logits.detach().cpu())

                elif self.args.dataset_name == 'pietrolesci/nli_fever':
                    self.cartography.update(batch['fid'], output.logits.detach().cpu())

                else:
                    raise Exception('PANICA')

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        error = zero_one_loss(y_true, y_pred, normalize = True)

        results = {
            'acc': accuracy_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred, average = 'macro'),
            'recall': recall_score(y_true, y_pred, average = 'macro'),
            'f1': f1_score(y_true, y_pred, average = 'macro'),
            'error': error,
        }

        if save:
            os.makedirs(f"results/{self.args.output_dir}", exist_ok = True)
            with open(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_results.json", 'wt') as f:
                json.dump(results, f)

        if save_characterization:
            os.makedirs(f"results/{self.args.output_dir}", exist_ok = True)
            with open(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_cartography.json", 'wt') as f:
                self.cartography.to_json(f)

        return results
