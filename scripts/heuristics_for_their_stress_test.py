from heuristics import length_missmatch, word_overlap, number_of_antonyms, misspelled_words, contains_negation
import pandas as pd
import tqdm
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

heuristic_functions = [
    # ('misspelled_words', misspelled_words),
    ('contains_negation', contains_negation),
    ('number_of_antonyms', number_of_antonyms),
    ('word_overlap', word_overlap),
    ('length_missmatch', length_missmatch),
]

df = pd.read_csv('./assets/stress-tests/stress_tests.csv')

new_df = df[['promptID', 'sentence1', 'sentence2', 'gold_label', 'kind']]
new_df = new_df.rename(columns = {'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})

for name, func in tqdm.tqdm(heuristic_functions):
    new_df[name] = new_df.parallel_apply(lambda row: func(row['premise'], row['hypothesis']), axis = 1)

new_df.to_csv('./assets/stress-tests/stress_tests_with_heuristics.csv', index = False)