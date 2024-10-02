import transformers
import torch
import pandas as pd
import os
from lib.dataset_extra import AcumenDataset

class OurStressTestNLIDataset(AcumenDataset):
    def __init__(self, args, dataset_name, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

        self.dataset_name = dataset_name

        current_folder_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_folder_path, '..', 'assets', f'{dataset_name}_final.csv')

        print("::: Loading OurNLI dataset :::")
        self.df = pd.read_csv(data_path)

        print(self.df['hardness:gaussian_mixture'].value_counts())

        old_length = len(self.df.index)

        # count rows with nans
        num_nans = self.df.isnull().sum(axis=1).sum()

        # remove nans in general
        self.df = self.df.dropna().reset_index(drop=True)
        print(f":::: Removed {num_nans} items with nans. ::::")

        print(f"::: [OurStressTestNLIDataset] Loaded {dataset_name}, and len = {len(self.df.index)}.:::")
        print("::: Removed {} items with no hardness level.:::".format(old_length - len(self.df.index)))

    def __len__(self):
        return len(self.df.index)

    @classmethod
    def train_dataloader(cls, args):
        raise NotImplementedError

    @classmethod
    def val_dataloader(cls, args, dataset_name, kind = 'validation'):
        dataset = cls(args = args, dataset_name = dataset_name, data_transforms = None, kind = kind)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
        )

    def __getitem__(self, idx):
        item = self.df.iloc[idx].to_dict()

        try:
            output = self.tokenizer.encode_plus(
                item['premise'], item['hypothesis'],
                padding = 'max_length',
                add_special_tokens = True,
                truncation = True,
                max_length = self.args.max_length,
                return_tensors = 'pt',
                return_token_type_ids = True,
            )
        except Exception as e:
            print(item)
            exit(-1)

        for column in item.keys():
            if column.startswith('heuristic:'):
                item[column] = torch.tensor(item[column])

        item['label'] = torch.tensor(item['true_label'])
        item['hardness'] = item['hardness:gaussian_mixture']

        item['input_ids'] = output['input_ids'][0]
        item['attention_mask'] = output['attention_mask'][0]
        item['token_type_ids'] = output['token_type_ids'][0]
        item['idx'] = torch.tensor(idx)

        return item
