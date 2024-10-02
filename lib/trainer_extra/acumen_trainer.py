import torch
import torch.nn as nn

class AcumenTrainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self._optimizer = None

    def configure_optimizers(self, lr = 0.03):
        if self._optimizer is not None:
            return self._optimizer

        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.05
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            },
        ]

        self._optimizer = torch.optim.AdamW(
            params,
            lr=lr,
        )

        return self._optimizer

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def validation_epoch_start(self, outputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, epoch = None):
        pass

    def training_batch_start(self, batch = None):
        pass

    def training_batch_end(self, batch = None):
        pass

    def training_epoch_start(self, epoch = None):
        pass

    def training_end(self):
        pass

    def training_start(self):
        pass
