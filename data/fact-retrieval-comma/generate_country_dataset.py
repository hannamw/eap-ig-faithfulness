# %%
import random 
from collections import defaultdict

import pandas as pd
from transformer_lens import HookedTransformer
from tqdm import tqdm 

from eap.utils import model2family
#%%
def create_dataset(model_name: str):
    family = model2family(model_name)
    model = HookedTransformer.from_pretrained(model_name)

    df = pd.read_csv('country_capitals.csv')

    length_city_dict = defaultdict(list)
    for city in df['capital']:
        city = city.strip()
        token_length = len(model.tokenizer(f' {city},').input_ids)
        length_city_dict[token_length].append(city)

    to_delete = []
    for fset, l in length_city_dict.items():
        if len(l) <= 1:
            print(f"found fset without proper alternatives {fset}: {l}")
            to_delete.append(fset)
            df = df[df['capital'] != l[0]]
    for dl in to_delete:
        del length_city_dict[dl]

    capital_country_mapping = {capital: country for country, capital in zip(df['country'], df['capital'])}

    dataset = {k:[] for k in ['clean', 'capital', 'country', 'country_idx', 'corrupted', 'corrupted_capital', 'corrupted_country', 'corrupted_country_idx']}

    for country, capital in tqdm(list(zip(df['country'], df['capital']))):
        capital_length = len(model.tokenizer(f' {capital},').input_ids)
        clean = f' {capital}, the capital of'
        country_idx = model.tokenizer(f' {country.strip()}').input_ids[0]

        valid_corrupted_capitals = length_city_dict[capital_length]
        assert len(valid_corrupted_capitals) > 1, f'{valid_corrupted_capitals}'
        corrupted_capital = capital
        while corrupted_capital == capital:
            corrupted_capital = random.choice(valid_corrupted_capitals)

        corrupted_country = capital_country_mapping[corrupted_capital]
        corrupted = f' {corrupted_capital}, the capital of'

        corrupted_country_idx = model.tokenizer(f' {corrupted_country.strip()}').input_ids[0]

        for k, v in zip(['clean', 'capital', 'country', 'country_idx', 'corrupted', 'corrupted_capital', 'corrupted_country', 'corrupted_country_idx'], [clean, capital, country, country_idx, corrupted, corrupted_capital, corrupted_country, corrupted_country_idx]):
            dataset[k].append(v)

    df2 = pd.DataFrame.from_dict(dataset)
    df2.to_csv(f'{family}.csv', index=False)

#%%
create_dataset('pythia-160m')
