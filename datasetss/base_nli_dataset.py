import datasets
import transformers
import torch
from lib.dataset_extra import AcumenDataset

class NLIDataset(AcumenDataset):
    def __init__(self, args, dataset_name = None, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)

        dataset_name = args.dataset_name if dataset_name is None else dataset_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

        if not self.args.save_characterization:
            new_kind = kind
            self.dataset = datasets.load_dataset(dataset_name, split = new_kind)
        else:
            new_kind = self.args.target_split

            if dataset_name == 'multi_nli' and new_kind == 'validation':
                new_kind = 'validation_matched'
                self.dataset = datasets.load_dataset(dataset_name, split = new_kind)

            elif dataset_name == 'anli' and new_kind == 'validation':
                new_kind = ['dev_r1', 'dev_r2', 'dev_r3']
                self.dataset = datasets.concatenate_datasets(datasets.load_dataset(dataset_name, split = new_kind))

            elif dataset_name == 'anli' and new_kind == 'test':
                new_kind = ['test_r1', 'test_r2', 'test_r3']
                self.dataset = datasets.concatenate_datasets(datasets.load_dataset(dataset_name, split = new_kind))

            else:
                self.dataset = datasets.load_dataset(dataset_name, split = new_kind)

        self.dataset_name = dataset_name

        if 'nli_fever' not in dataset_name:
            self.dataset = self.dataset.filter(lambda x: ((x['label'] != -1) and (len(x['premise']) > 0) and (len(x['hypothesis']) > 0)))

        print(f"::: [NLIDataset] Loaded {dataset_name} with new_kind = {new_kind}, and len = {len(self.dataset)} :::")

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def train_dataloader(cls, args):
        if args.dataset_name == 'anli':
            kind = 'train_' + args.anli_split
        else:
            kind = 'train'

        dataset = cls(args = args, kind = kind)

        return torch.utils.data.DataLoader(
            dataset,
            num_workers = 4 ,
            pin_memory = True,
            batch_size = args.batch_size
        )

    @classmethod
    def val_dataloader(cls, args, dataset_name = None, kind = 'validation'):
        if dataset_name is None:
            dataset_name = args.dataset_name

        if dataset_name == 'multi_nli':
            new_kind = 'validation_matched'
        elif dataset_name == 'anli':
            new_kind = 'dev_' + args.anli_split
        elif 'nli_fever' in dataset_name:
            new_kind = 'dev'
        else:
            new_kind = kind

        dataset = cls(args = args, data_transforms = None, dataset_name = dataset_name, kind = new_kind)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
        )

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.args.use_hypo:
            output = self.tokenizer.encode_plus(
                item['hypothesis'],
                padding = 'max_length',
                add_special_tokens = True,
                truncation = True,
                max_length = self.args.max_length,
                return_tensors = 'pt',
                return_token_type_ids = True,
            )
        else:
            output = self.tokenizer.encode_plus(
                item['premise'], item['hypothesis'],
                padding = 'max_length',
                add_special_tokens = True,
                truncation = True,
                max_length = self.args.max_length,
                return_tensors = 'pt',
                return_token_type_ids = True,
            )

        item['label'] = torch.tensor(item['label'])
        item['input_ids'] = output['input_ids'][0]
        item['attention_mask'] = output['attention_mask'][0]
        item['token_type_ids'] = output['token_type_ids'][0]
        item['idx'] = torch.tensor(idx)

        if 'nli_fever' in self.dataset_name:
            del item['verifiable']

        return item
