#%%
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#%%
model_name = 'gpt2'
model_name_noslash = model_name.split('/')[-1]

display_name_dict = {'ioi': 'IOI', 'greater-than': "Greater-Than", 'greater-than-price': "Greater-Than (Price)", 'greater-than-sequence': "Greater-Than (Sequence)", 'gender-bias': 'Gender-Bias', 'sva': 'SVA', 'fact-retrieval': 'Country-Capital', 'hypernymy': 'Hypernymy'}
def display_name(task):
    if '-comma' in task:
        task = task[:-6]
    return display_name_dict[task] 

prob_diff_tasks = ['greater-than', 'fact-retrieval', 'sva']
def task_to_metric(task):
    if any(ptask in task for ptask in prob_diff_tasks):
        return 'prob_diff'
    return 'logit_diff'

def metric_to_name(metric):
    if 'real' in metric:
        return 'activation-patching' + metric[4:]
    else:
        return metric
    
prob_diff_tasks = ['greater-than', 'fact-retrieval', 'sva']
def task_to_metric(task: str):
    if any(ptask in task for ptask in prob_diff_tasks):
        return 'prob_diff'
    return 'logit_diff'

kl_ig_tasks = {'hypernymy-comma', 'sva', 'fact-retrieval-comma', 'greater-than-price', 'greater-than-sequence'}
def task_to_path(task: str):
    if task in kl_ig_tasks:
        return f'graphs/{model_name_noslash}/real_test/{task}_kl_ig.json'
    else:
        return f'graphs/{model_name_noslash}/real_test/{task}_{task_to_metric(task)}_real.json'
real_task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma', 'greater-than-price', 'greater-than-sequence']
display_names = [display_name(name) for name in real_task_names]
#%%
df = pd.read_csv(f'{model_name_noslash}/csv/faithfulness.csv')
z = df.to_numpy()
baseline = z[:, -2:-1]
corrupted_baseline = z[:, -1:]
z = z[:, :-2]
normalized = (z - corrupted_baseline)/(baseline - corrupted_baseline)
#%%
heat = go.Heatmap(z=normalized,
                x=display_names,
                y=display_names,
    
                colorbar={"title": 'faithfulness'},
                xgap=1, ygap=1,
                colorscale='Blues',
                colorbar_thickness=20,
                colorbar_ticklen=3,
                )
layout = go.Layout(title_text='Faithfulness', title_x=0.5, 
                width=600, height=600,
                xaxis = {'title': 'Tested on'},
                yaxis = {'title': 'Circuit From'},
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange='reversed')

fig=go.Figure(data=[heat], layout=layout)   
fig.show()     
# %%
