#%%
import pandas as pd 

from transformers import AutoTokenizer

from ioi_dataset import IOIDataset
from eap.utils import model2family
# %%
model_name = 'gpt2'
model_name_noslash = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
ds = IOIDataset('mixed', N=1000, tokenizer=tokenizer)
# %%
abc_dataset = (  # TODO seeded
    ds.gen_flipped_prompts(("S2", "RAND"))
)
abb_dataset = (  # TODO seeded
    ds.gen_flipped_prompts(("S2", "IO"))
)

#%%
d = {'clean': [], 'corrupted': [], 'corrupted_hard': [], 'correct_idx': [], 'incorrect_idx': []}
for i in range(len(ds)):
    clean = ' '.join(ds.sentences[i].split()[:-1])
    corrupted = ' '.join(abc_dataset.sentences[i].split()[:-1])
    corrupted_hard = ' '.join(abb_dataset.sentences[i].split()[:-1])
    correct = ds.toks[i, ds.word_idx['IO'][i]].item()
    incorrect = ds.toks[i, ds.word_idx['S'][i]].item()
    d['clean'].append(clean)
    d['corrupted'].append(corrupted)
    d['corrupted_hard'].append(corrupted_hard)
    d['correct_idx'].append(correct)
    d['incorrect_idx'].append(incorrect)

df = pd.DataFrame.from_dict(d)
df = df.sample(frac=1)
df.to_csv(f'{model2family(model_name)}.csv')
# %%
