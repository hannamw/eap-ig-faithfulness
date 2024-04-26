#%%
from functools import partial 
from argparse import ArgumentParser

import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm


from eap.graph import Graph
from eap.attribute_mem import attribute
from eap.evaluate_graph import evaluate_graph, evaluate_baseline
from dataset import EAPDataset
from metrics import get_metric
#%%
parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-t', '--task', type=str, required=True)
parser.add_argument('--head', type=int, default=None)
parser.add_argument('--batch_size', type=int, required=True)
args = parser.parse_args()
model_name = args.model 
task = args.task

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
metric_names = ['logit_diff', 'prob_diff', 'kl', 'js']
ds = EAPDataset(task, model_name)
np.random.seed(42)
if args.head is not None:
    ds.head(args.head)

dataloader_patching = ds.to_dataloader(args.batch_size)
metrics = [partial(get_metric(metric_name, task, model=model), mean=False, loss=False) for metric_name in metric_names]
# %%
baselines = torch.stack(evaluate_baseline(model, dataloader_patching, metrics)).mean(-1)
#%%
# Instantiate a graph with a model
gs = [Graph.from_model(model) for _ in metrics]

#%%
for edge_name, edge in tqdm(list(gs[0].edges.items())):
    edge.in_graph = False
    performances = torch.stack(evaluate_graph(model, gs[0], dataloader_patching, metrics, quiet=True)).mean(-1)
    differences = (performances - baselines).tolist()
    for g, diff in zip(gs, differences):
        g.edges[edge_name].score = diff
    edge.in_graph = True

for g, metric_name in zip(gs, metric_names):
    g.to_json(f'graphs/{model_name_noslash}/real_test/{task}_{metric_name}_real.json')


