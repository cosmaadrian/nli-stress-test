import transformers
import torch
import pandas as pd
import os
from lib.dataset_extra import AcumenDataset

class OurNLIDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        assert kind == 'train', 'OurNLIDataset is only for training'

        super().__init__(args = args, kind = kind, data_transforms = data_transforms)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

        current_folder_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_folder_path, '..', 'assets', f'{args.dataset_name}-train_final.csv')

        print("::: Loading OurNLI dataset :::")
        self.df = pd.read_csv(data_path)

        old_length = len(self.df.index)

        if self.args.hardness_key == 'hardness:datamaps':
            percentiles_std = self.df['both_std_conf'].quantile([0.66])
            self.df = self.df[self.df['both_std_conf'] > percentiles_std[0.66]].reset_index(drop=True)
        elif self.args.hardness_key == 'hardness:aum':
            percentiles_aum = self.df['both_average_margin'].quantile([0.33, 0.66])
            # take the middle
            self.df = self.df[(self.df['both_average_margin'] > percentiles_aum[0.33]) & (self.df['both_average_margin'] < percentiles_aum[0.66])].reset_index(drop=True)
        else:
            print(self.df[self.args.hardness_key].value_counts())
            # get only items with some hardness level
            self.df = self.df[self.df[self.args.hardness_key].isin(args.hardness)].reset_index(drop=True)

        # count rows with nans
        num_nans = self.df.isnull().sum(axis=1).sum()

        # remove nans in general
        self.df = self.df.dropna().reset_index(drop=True)
        print(f":::: Removed {num_nans} items with nans. ::::")

        print(f"::: [OurNLIDataset] Loaded {args.dataset} {args.dataset_name}, and len = {len(self.df.index)} with hardness {args.hardness}.:::")
        print(f"::: Removed {old_length - len(self.df.index)} items with no hardness level. ({old_length} / {len(self.df.index)} - {len(self.df.index) / old_length} kept):::")

        if self.args.do_sample:
            self.df = self.df.sample(frac = self.args.sample_percent).reset_index(drop = True)
            print("::: [OurNLIDataset] Sampled {} items. :::".format(len(self.df.index)))
            print('::: [OurNLIDataset] Hardness distribution after sampling :::')
            print(self.df[self.args.hardness_key].value_counts())

    def __len__(self):
        return len(self.df.index)

    @classmethod
    def train_dataloader(cls, args):
        dataset = cls(args = args, kind = 'train')

        return torch.utils.data.DataLoader(
            dataset,
            num_workers = 4 ,
            pin_memory = True,
            batch_size = args.batch_size
        )

    @classmethod
    def val_dataloader(cls, args, kind = 'validation'):
        raise NotImplementedError

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

        item['label'] = torch.tensor(item['true_label'])

        item['input_ids'] = output['input_ids'][0]
        item['attention_mask'] = output['attention_mask'][0]
        item['token_type_ids'] = output['token_type_ids'][0]
        item['idx'] = torch.tensor(idx)

        return item
