#%%
from argparse import ArgumentParser
from functools import partial 
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
def display_name(task):
    if '-comma' in task:
        task = task[-6:]
    return task 

prob_diff_tasks = ['greater-than', 'fact-retrieval', 'sva']
def task_to_metric(task):
    if any(ptask in task for ptask in prob_diff_tasks):
        return 'prob_diff'
    return 'logit_diff'

#%%
model = 'gpt2'
task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma']
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6))
for task_name, ax in zip(task_names, axs.flat):
    df = pd.read_csv(f'{model}/csv/{task_name}.csv')
    ax.plot(df['edges'], df['baseline'], linestyle='dotted', label='clean baseline')
    ax.plot(df['edges'], df['corrupted_baseline'], linestyle='dotted', label='corrupted baseline')
    ax.plot(df['edges_EAP'], df['loss_EAP'], label='EAP')
    ax.plot(df['edges_EAP-IG'], df['loss_EAP-IG'], label='EAP-IG')
    ax.plot(df['edges_EAP-IG-KL'], df['loss_EAP-IG-KL'], label='EAP-IG-KL')
    ax.plot(df['edges_EAP-IG-JS'], df['loss_EAP-IG-JS'], label='EAP-IG-JS')
    #ax.legend()
    #ax.set_xlabel(f'Edges included (/32491)')
    ax.set_ylabel(f'{task_to_metric(task_name)}')
    ax.set_title(f'{task_name}')
handles, labels = ax.get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)
fig.text(0.5, 0.00, f'Edges included (/32491)', ha='center')
fig.tight_layout()
fig.savefig(f'{model}/png/all_plots.png', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.savefig(f'{model}/pdf/all_plots.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.show()
# %%
