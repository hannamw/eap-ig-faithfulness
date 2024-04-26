#%%
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from greater_than_price_dataset import YearDataset, get_valid_years
from eap.utils import model2family
#%%
def create_dataset(model_name, N):
    items = [
    "gem",
    "necklace",
    "watch",
    "ring",
    "suitcase",
    "scarf",
    "suit",
    "shirt",
    "sweater",
    "dress",
    "fridge",
    "TV",
    "bed",
    "bike",
    "lamp",
    "table",
    "chair",
    "painting",
    "sculpture",
    "plant",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    items = [item for item in items if len(tokenizer(f' {item}')['input_ids']) == 1]
    ds = YearDataset(get_valid_years(tokenizer, 1100, 1800), N, items, tokenizer)

    random_order = np.random.permutation(N)
    def apply_order(xs):
        return [xs[i] for i in random_order]
    d = {'clean': apply_order(ds.good_sentences), 'corrupted': apply_order(ds.bad_sentences),  'correct_idx': apply_order(ds.years_YY.tolist())}

    df = pd.DataFrame.from_dict(d)
    df = df.sample(frac=1)
    df.to_csv(f'{model2family(model_name)}.csv', index=False)

#%%
if __name__ == '__main__':
    model_name = 'gpt2'
    N = 1000
    create_dataset(model_name, N)
# %%
