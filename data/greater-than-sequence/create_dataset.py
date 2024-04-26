#%%
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch

from greater_than_dataset import get_valid_years
from eap.utils import model2family
#%%
def create_dataset(model_name, N):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    years = []
    centuries = torch.arange(10,19) * 100
    years_XX00 = centuries[torch.randint(len(centuries), (N,))]
    years_XX = years_XX00 // 100
    years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)

    for XX00 in years_XX00:
        sample_space = years_to_sample_from[(years_to_sample_from >= XX00) & (years_to_sample_from < XX00+100)]
        current = sample_space[torch.randint(0, sample_space.size(0),(1,))]
        current_years = [current]
        for i in range(4):
            sample_space = years_to_sample_from[(years_to_sample_from > current) & (years_to_sample_from < current+30)]
            if len(sample_space) == 0:
                break
            current = sample_space[torch.randint(0, sample_space.size(0),(1,))]
            current_years.append(current)
        if len(current_years) == 5:
            years.append(torch.cat(current_years))

    years = torch.stack(years)
    years_YY = years % 100
    final_century = (years[:, -1] // 100)
    year_has_valid_answer = torch.tensor([len(years_to_sample_from[(years_to_sample_from > years[i, -1]) & (years_to_sample_from < (final_century[i]+1) * 100) & (years[i, -2]// 100 < 18)]) > 0 for i in range(years.size(0))])

    years = years[year_has_valid_answer]
    years_YY = years % 100
    final_century = (years[:, -1] // 100)
    final_century_corrupted = (years[:, -2] // 100) + 1

    sentences = [f'{str(y.tolist())[1:-1]}, {XX}' for y, XX in zip(years, final_century)]
    sentences_01 = [f'{str(y.tolist())[1:-5]}{XX}01, {XX}' for y, XX in zip(years, final_century_corrupted)]
    last_two_digits = years_YY[:, -1]

    d = {'clean': sentences, 'corrupted': sentences_01,  'correct_idx': last_two_digits}

    df = pd.DataFrame.from_dict(d)
    df = df.sample(frac=1)
    df.to_csv(f'{model2family(model_name)}.csv', index=False)

#%%
if __name__ == '__main__':
    model_name = 'gpt2'
    N = 1000
    create_dataset(model_name, N)

# %%
