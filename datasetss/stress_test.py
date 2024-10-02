import transformers
import torch
import pandas as pd
import os
from lib.dataset_extra import AcumenDataset

name2label = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}

label2name = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction',
}

class StressTestNLIDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        assert kind != 'train', 'StressTestNLIDataset is only for validation and test'

        super().__init__(args = args, kind = kind, data_transforms = data_transforms)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

        current_folder_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_folder_path, '..', 'assets', 'stress-tests', 'stress_tests.csv')

        self.df = pd.read_csv(data_path)

        print(f"::: Loaded 'stress-tests.csv', and len = {len(self.df)}.:::")

    def __len__(self):
        return len(self.df.index)

    @classmethod
    def train_dataloader(cls, args):
        raise NotImplementedError

    @classmethod
    def val_dataloader(cls, args, kind = 'validation'):
        dataset = cls(args = args, data_transforms = None, kind = kind)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
        )

    def __getitem__(self, idx):
        old_item = self.df.iloc[idx].to_dict()

        item = dict()

        output = self.tokenizer.encode_plus(
            old_item['sentence1'], old_item['sentence2'],
            padding = 'max_length',
            add_special_tokens = True,
            truncation = True,
            max_length = self.args.max_length,
            return_tensors = 'pt',
            return_token_type_ids = True,
        )
        # item['label'] = torch.tensor(int(item['label']))

        item['label'] = torch.tensor(name2label[old_item['gold_label']])
        item['kind'] = old_item['kind']

        item['input_ids'] = output['input_ids'][0]
        item['attention_mask'] = output['attention_mask'][0]
        item['token_type_ids'] = output['token_type_ids'][0]
        item['idx'] = torch.tensor(idx)

        return item
