#%%
from argparse import ArgumentParser
from functools import partial

from tqdm import tqdm
import pandas as pd
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate_graph import evaluate_graph, evaluate_baseline

from dataset import EAPDataset
from metrics import get_metric
# %%
parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('--head', type=int, default=None)
parser.add_argument('--batch_size', type=int, required=True)

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

# %%
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
real_graphs = [Graph.from_json(task_to_path(task)) for task in real_task_names]
graphs_names = real_task_names
graphs = real_graphs
edge_thresholds = [800, 200, 200, 400, 500, 1600, 300, 200]
#%%
d = {}
for task1, graph, thresh in tqdm(zip(real_task_names, real_graphs, edge_thresholds), total=len(real_graphs)):
    perfs = []
    graph.apply_greedy(thresh, absolute=True)
    for task2 in real_task_names:
        dataset = EAPDataset(task2, model_name)
        if args.head is not None:
            dataset.head(args.head)
        dataloader = dataset.to_dataloader(args.batch_size)
        metric_name = task_to_metric(task2)
        metric = get_metric(metric_name, task2, model=model)
        perf = evaluate_graph(model, graph, dataloader, partial(metric, mean=False, loss=False),quiet=True).mean().item()
        perfs.append(perf)

    d[task1] = perfs

baselines, corrupted_baselines = [], []
for task2 in real_task_names:
    dataset = EAPDataset(task2, model_name)
    if args.head is not None:
        dataset.head(args.head)
    dataloader = dataset.to_dataloader(args.batch_size)
    metric_name = task_to_metric(task2)
    metric = get_metric(metric_name, task2, model=model)
    baseline = evaluate_baseline(model, dataloader, partial(metric,mean=False, loss=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, dataloader, partial(metric,mean=False, loss=False), run_corrupted=True).mean().item()

    baselines.append(baseline)
    corrupted_baselines.append(corrupted_baseline)

d['baseline'] = baselines
d['corrupted_baseline'] = corrupted_baselines
df = pd.DataFrame.from_dict(d)
df.to_csv(f'results/cross-task/{model_name_noslash}/csv/faithfulness.csv', index=False)