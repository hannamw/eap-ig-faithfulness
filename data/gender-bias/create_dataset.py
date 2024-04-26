#%%
import pandas as pd
from transformers import AutoTokenizer

from eap.utils import model2family
from professions import ProfessionsData
#%%
def create_dataset(model_name):
    data = ProfessionsData()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    she_token = tokenizer(' she', add_special_tokens=False).input_ids[0]
    he_token = tokenizer(' he', add_special_tokens=False).input_ids[0]
    d = {'clean': [], 'corrupted': [], 'clean_answer_idx':[], 'corrupted_answer_idx':[], 'label':[]}
    for clean, corrupted, label, _ in data.data:
        clean_len = len(tokenizer(clean).input_ids)
        corrupted_len = len(tokenizer(corrupted).input_ids)
        if clean_len != corrupted_len:
            continue 
        d['clean'].append(clean)
        d['corrupted'].append(corrupted)
        d['label'].append(label)
        clean_idx = he_token if label else she_token 
        corrupted_idx = she_token if label else he_token
        d['clean_answer_idx'].append(clean_idx)
        d['corrupted_answer_idx'].append(corrupted_idx)
    df = pd.DataFrame.from_dict(d)
    shortest_label_count = min((df['label'] == 0).to_numpy().sum(), (df['label'] == 1).to_numpy().sum())
    df = df.sample(frac=1)
    df0 = df[df['label'] == 0].head(shortest_label_count)
    df1 = df[df['label'] == 1].head(shortest_label_count)
    df = pd.concat([df0, df1])
    df = df.sample(frac=1)

    df.to_csv(f'{model2family(model_name)}.csv', index=False)

if __name__ == '__main__':
    model_name = 'EleutherAI/pythia-160m'
    create_dataset(model_name)


# %%
