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

from eap.graph import Graph
from eap.attribute_mem import attribute
from eap.evaluate_graph import evaluate_graph, evaluate_baseline

from dataset import EAPDataset
from metrics import get_metric
#%%
parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--metric', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--eval_batch_size', type=int, required=True)
parser.add_argument('--end_proportion', type=float, default=0.04)
parser.add_argument('--n_steps', type=int, default=10)
parser.add_argument('--head', type=int, default=1000)
args = parser.parse_args()

model_name = args.model
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
#%%
labels = ['EAP', 'EAP-IG', 'EAP-IG-KL']
task = args.task
task_metric_name = args.metric
ds = EAPDataset(task, model_name)
ds.head(args.head)

batch_size = args.batch_size #48
eval_batch_size = args.eval_batch_size
dataloader = ds.to_dataloader(batch_size)
eval_dataloader = ds.to_dataloader(eval_batch_size)
task_metric = get_metric(task_metric_name, task, model=model)
kl_div = get_metric('kl_divergence', task, model=model)

# %%
baseline = evaluate_baseline(model, dataloader, partial(task_metric, mean=False, loss=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(task_metric, mean=False, loss=False), run_corrupted=True).mean().item()
#%%
# Instantiate a graph with a model
g1 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute(model, g1, dataloader, partial(task_metric, mean=True, loss=True))
Path(f'graphs/{model_name_noslash}').mkdir(exist_ok=True, parents=True)
g1.to_json(f'graphs/{model_name_noslash}/{task}_vanilla.json')
#%%
# Instantiate a graph with a model
g2 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute(model, g2, dataloader, partial(task_metric, mean=True, loss=True), integrated_gradients=5)
#attribute(model, g, clean, corrupted, labels, kl_div, integrated_gradients=30)
g2.to_json(f'graphs/{model_name_noslash}/{task}_task.json')
#%%
# Instantiate a graph with a model
g3 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute(model, g3, dataloader, partial(kl_div, mean=True, loss=True), integrated_gradients=5)
g3.to_json(f'graphs/{model_name_noslash}/{task}_kl.json')

# %%
gs = [g1, g2, g3]
n_edges = []
results = []
s = 100
e = int(len(g1.edges) * args.end_proportion) 
step = (e - s) // args.n_steps
steps = list(range(s,e+1, step))
with tqdm(total=len(gs)*len(steps)) as pbar:
    for i in steps:
        n_edge = []
        result = []
        for graph in gs:
            graph.apply_greedy(i, absolute=True)
            graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
            n = graph.count_included_edges()
            r = evaluate_graph(model, graph, eval_dataloader, partial(task_metric, mean=False, loss=False), quiet=True)
            n_edge.append(n)
            result.append(r.mean().item())
            pbar.update(1)
        n_edges.append(n_edge)
        results.append(result)

n_edges = np.array(n_edges)
results = np.array(results)
#%%
d = {'baseline':[baseline] * len(steps), 
     'corrupted_baseline':[corrupted_baseline] * len(steps),
     'edges': steps,
     'total_edges':[len(g1.edges)] * len(steps)}

for i, label in enumerate(labels):
    d[f'edges_{label}'] = n_edges[:, i].tolist()
    d[f'loss_{label}'] = results[:, i].tolist()
df = pd.DataFrame.from_dict(d)
Path(f'results/pareto/{model_name_noslash}/csv').mkdir(exist_ok=True, parents=True)
df.to_csv(f'results/pareto/{model_name_noslash}/csv/{task}.csv', index=False)
# %%
fig, ax = plt.subplots()
ax.plot(steps, [baseline] * len(steps), linestyle='dotted', label='clean baseline')
ax.plot(steps, [corrupted_baseline] * len(steps), linestyle='dotted', label='corrupted baseline')

for i, label in enumerate(labels):
    ax.plot(n_edges[:, i], results[:, i], label=label)
ax.legend()
ax.set_xlabel(f'Edges included (/{len(gs[0].edges)})')
ax.set_ylabel(f'{task_metric_name}')
ax.set_title(f'{task} EAP vs. EAP-IG ({model_name_noslash})')
fig.show()

Path(f'results/pareto/{model_name_noslash}/png').mkdir(exist_ok=True, parents=True)
Path(f'results/pareto/{model_name_noslash}/pdf').mkdir(exist_ok=True, parents=True)
fig.savefig(f'results/pareto/{model_name_noslash}/png/{task}.png')
fig.savefig(f'results/pareto/{model_name_noslash}/pdf/{task}.pdf')

# %%
