import pandas as pd
import transformers
import json
import numpy as np
from scipy.special import softmax
import tqdm

from lib.arg_utils import define_args
from lib.utils import load_model_by_name
from lib import nomenclature
import torch
import lib

from heuristics import length_missmatch, word_overlap, number_of_antonyms, misspelled_words, contains_negation

args = define_args(
    extra_args = [
        ('--output_dir', {'default': 'test', 'type': str, 'required': False}),
    ]
)

dataset = nomenclature.DATASETS[args.dataset](args = args, kind = args.target_split)

file_both = f"results/{args.output_dir}/{args.group}:{args.name}-both_cartography.json"
file_hypo = f"results/{args.output_dir}/{args.group}:{args.name}-hypo_cartography.json"

with open(file_both, 'rt') as f:
    data_both = json.load(f)

with open(file_hypo, 'rt') as f:
    data_hypo = json.load(f)

heuristic_functions = [
    ('length_missmatch', length_missmatch),
    ('word_overlap', word_overlap),
    ('number_of_antonyms', number_of_antonyms),
    ('misspelled_words', misspelled_words),
    ('contains_negation', contains_negation),
]

new_data = []
model_name = 'roberta-base' if 'roberta-base' in args.model_name else 'deberta'

if 'fever' in args.dataset_name:
    state_dict = load_model_by_name(f'{args.group}:fever-{model_name}-train-both2')
else:
    state_dict = load_model_by_name(f'{args.group}:{args.dataset_name}-{model_name}-train-both')

model = nomenclature.MODELS['transformer'](args = args)

model.load_state_dict(state_dict)
model.eval()
model.train(False)
model.cuda()

if 'fever' in args.dataset_name:
    state_dict = load_model_by_name(f'{args.group}:fever-{model_name}-train-hypo')
else:
    state_dict = load_model_by_name(f'{args.group}:{args.dataset_name}-{model_name}-train-hypo')

model_hypo = nomenclature.MODELS['transformer'](args = args)

model_hypo.load_state_dict(state_dict)
model_hypo.eval()
model_hypo.train(False)
model_hypo.cuda()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

def characterize(data, sample):
    logit_list = data
    true_label_idx = sample['label']

    logit_list_np = np.array(logit_list)
    conf_softmax = softmax(logit_list_np, axis=1)

    max_idx = conf_softmax.argmax(axis=1)

    conf_softmax_masked = conf_softmax.copy()
    conf_softmax_masked[np.arange(len(conf_softmax)), true_label_idx] = -1e6

    second_max_idx = conf_softmax_masked.argsort(axis=1)[:, -1].ravel()

    margin = logit_list_np[np.arange(len(logit_list_np)), true_label_idx] - logit_list_np[np.arange(len(logit_list_np)), second_max_idx]
    average_margin = margin.mean()

    mean_conf = conf_softmax[:, true_label_idx].mean()
    std_conf = conf_softmax[:, true_label_idx].std()

    average_correctness = (max_idx == true_label_idx).mean()

    return {
        'mean_conf': mean_conf,
        'std_conf': std_conf,
        'average_margin': average_margin,
        'final_pred_label': max_idx[-1],
        'average_correctness': average_correctness,
    }

@torch.no_grad()
def predict(sample):
    output = tokenizer.encode_plus(
        sample['premise'], sample['hypothesis'],
        padding = 'max_length',
        add_special_tokens = True,
        truncation = True,
        max_length = args.max_length,
        return_tensors = 'pt',
        return_token_type_ids = True,
    )

    input_ids = output['input_ids'].to(lib.device)
    attention_mask = output['attention_mask'].to(lib.device)
    token_type_ids = output['token_type_ids'].to(lib.device)

    model_prediction = model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
    model_prediction = model_prediction['nli']

    return model_prediction

@torch.no_grad()
def predict_hypo(sample):
    output = tokenizer.encode_plus(
        sample['hypothesis'],
        padding = 'max_length',
        add_special_tokens = True,
        truncation = True,
        max_length = args.max_length,
        return_tensors = 'pt',
        return_token_type_ids = True,
    )

    input_ids = output['input_ids'].to(lib.device)
    attention_mask = output['attention_mask'].to(lib.device)
    token_type_ids = output['token_type_ids'].to(lib.device)

    model_prediction = model_hypo({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
    model_prediction = model_prediction['nli']

    return model_prediction

for i, sample in enumerate(tqdm.tqdm(dataset.dataset)):
    if args.dataset_name == 'anli':
        key = sample['uid']
    elif args.dataset_name == 'snli':
        key = i
    elif args.dataset_name == 'multi_nli':
        key = sample['pairID']
    elif 'nli_fever' in args.dataset_name:
        key = sample['fid']

    both_characterization = characterize(data_both[str(key)], sample)
    hypo_characterization = characterize(data_hypo[str(key)], sample)

    model_prediction = predict(sample)
    predicted_label = model_prediction.labels[0].item()

    model_prediction_hypo = predict_hypo(sample)
    predicted_label_hypo = model_prediction_hypo.labels[0].item()

    new_data.append({
        'idx': key,
        'premise': sample['premise'],
        'hypothesis': sample['hypothesis'],
        'true_label': sample['label'],
        f'{model_name}-prediction': predicted_label,
        f'{model_name}-hypo-prediction': predicted_label_hypo,
        **{'hypo_' + k: v for k, v in hypo_characterization.items()},
        **{'both_' + k: v for k, v in both_characterization.items()},
        **{'heuristic:' + name: func(sample['premise'], sample['hypothesis']) for name, func in heuristic_functions}
    })

df = pd.DataFrame(new_data)
df.to_csv(f"results/{args.output_dir}/{args.group}:{args.name}-stats2.csv", index=False)