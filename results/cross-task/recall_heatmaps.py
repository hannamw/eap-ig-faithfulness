#%%
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

kl_ig_tasks = {'hypernymy-comma', 'sva', 'fact-retrieval-comma', 'greater-than-price', 'greater-than-sequence'}
def task_to_path(task: str):
    if task in kl_ig_tasks:
        return f'graphs/{model_name_noslash}/real_test/{task}_kl_ig.json'
    else:
        return f'graphs/{model_name_noslash}/real_test/{task}_{task_to_metric(task)}_real.json'
    
def make_graph_hm(z, graphs_names):
    heat = go.Heatmap(z=z,
                  x=graphs_names,
                  y=graphs_names,
                  colorbar={"title": 'IoU'},
                  xgap=1, ygap=1,
                  colorscale='Blues',
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                   )
    #layout = go.Layout(title_text=title, title_x=0.5, 
    #                width=600, height=600,
    #                xaxis_showgrid=False,
    #                yaxis_showgrid=False,
    #                yaxis_autorange='reversed')
    
    #fig=go.Figure(data=[heat], layout=layout)        
    return heat

real_task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma', 'greater-than-price', 'greater-than-sequence']
display_names = [display_name(name) for name in real_task_names]
#%%
df = pd.read_csv(f'{model_name_noslash}/csv/faithfulness.csv')
z = df.to_numpy()
baseline = z[:, -2:-1] # z[:, -2:-1]
corrupted_baseline = z[:, -1:] # z[:, -1:]
z = z[:, :-2]
normalized = (z - corrupted_baseline)/(baseline - corrupted_baseline)
task_circuit_baseline = z[np.arange(len(z)), np.arange(len(z))].reshape(-1,1)
task_circuit_normalized = (z - corrupted_baseline)/(task_circuit_baseline - corrupted_baseline)


node_iou_df = pd.read_csv('../jaccard/csv/node_overlap.csv')
edge_iou_df = pd.read_csv('../jaccard/csv/edge_overlap.csv')
node_iou_heatmap = make_graph_hm(node_iou_df.to_numpy(), display_names)
edge_iou_heatmap = make_graph_hm(edge_iou_df.to_numpy(), display_names)

node_z = node_iou_df.to_numpy()
edge_z = edge_iou_df.to_numpy()

task_circuit_normalized_flat = task_circuit_normalized.reshape(-1)
node_z_flat = node_z.reshape(-1)
edge_z_flat = edge_z.reshape(-1)
task_circuit_normalized_flat = task_circuit_normalized.reshape(-1)
both = np.stack([node_z_flat, edge_z_flat]).transpose(1,0)

print(pearsonr(node_z_flat, task_circuit_normalized_flat))
print(pearsonr(edge_z_flat, task_circuit_normalized_flat))
lr1, lr2, lr3 = LinearRegression(), LinearRegression(), LinearRegression()
lr1.fit(node_z_flat.reshape(-1,1), task_circuit_normalized_flat)
lr2.fit(edge_z_flat.reshape(-1,1), task_circuit_normalized_flat)
lr3.fit(both, task_circuit_normalized_flat)
print(lr1.score(node_z_flat.reshape(-1,1), task_circuit_normalized_flat))
print(lr2.score(edge_z_flat.reshape(-1,1), task_circuit_normalized_flat))
print(lr3.score(both, task_circuit_normalized_flat))

# %%
heat1 = go.Heatmap(z=node_z,
                  x=display_names,
                  y=display_names,
                  #colorbar={"title": 'IoU'},
                  xgap=1, ygap=1,
                  coloraxis="coloraxis1"
                   )

heat2 = go.Heatmap(z=edge_z,
                  x=display_names,
                  y=display_names,
                  #colorbar={"title": 'IoU'},
                  xgap=1, ygap=1,
                  coloraxis="coloraxis1"
                   )

heat3 = go.Heatmap(z=task_circuit_normalized,
                x=display_names,
                y=display_names,
                #colorbar={"title": 'faithfulness'},
                xgap=1, ygap=1,
                coloraxis="coloraxis1"
                )
#%%
fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=["Node Recall", "Edge Recall", "Cross-Task Faithfulness"],
    horizontal_spacing=0.03,
    vertical_spacing=0.1,
)
fig.add_trace(heat1, row=1, col=1)
fig.add_trace(heat2, row=1, col=2)
fig.add_trace(heat3, row=1, col=3)

fig.update_layout(
    width=1400,
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    coloraxis1=dict(
        colorscale="Blues",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title_side="right",
        cmin=0.0,
        cmax=1.0
    ),
)
fig.update_yaxes(autorange="reversed")
fig.write_image(f'{model_name_noslash}/png/heatmaps_recall_renormalized.png')
fig.write_image(f'{model_name_noslash}/pdf/heatmaps_recall_renormalized.pdf')
fig.show()
# %%
# %%

