import pandas as pd
import numpy as np
import glob
import json


files = glob.glob('../assets/stress-tests/Stress Tests/**/*.jsonl')

all_dfs = []

for file in files:
    kind = file.split('/')[-2]
    df = pd.read_json(file, lines=True)
    df['kind'] = kind
    all_dfs.append(df)

df = pd.concat(all_dfs)
df.to_csv('../assets/stress-tests/stress_tests.csv', index=False)