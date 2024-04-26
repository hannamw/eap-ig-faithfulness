#%%
from argparse import ArgumentParser
from pathlib import Path
from functools import partial 
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph 
from eap.attribute_mem import attribute
from eap.evaluate_graph import evaluate_graph, evaluate_baseline
from dataset import EAPDataset
from metrics import get_metric
#%%
def attribute_graphs(model_name, task, metric_name, batch_size):
    model_name_noslash = model_name.split('/')[-1]
    model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device='cuda',
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    graph = Graph.from_model(model)
    metric = get_metric(metric_name, task=task, model=model)
    dataset = EAPDataset(task=task, model_name=model_name)
    dataloader = dataset.to_dataloader(batch_size)

    if ('kl' not in metric_name) and ('js' not in metric_name):
        attribute(model, graph, dataloader, partial(metric, mean=True, loss=True), integrated_gradients=None, quiet=True)
        graph.to_json(f'graphs/{model_name_noslash}/ig_test/{task}_{metric_name}_vanilla.json')

    for n_iter in tqdm([2,3,5,7, 10, 15, 20, 30, 50]):
        attribute(model, graph, dataloader, partial(metric, mean=True, loss=True), integrated_gradients=n_iter, quiet=True)
        graph.to_json(f'graphs/{model_name_noslash}/ig_test/{task}_{metric_name}_{n_iter}.json')

def evaluate_graphs(model_name, task, attribution_metric_name, evaluation_metric_name, batch_size, end=1001):
    n_edges_list = list(range(100, end, 100))
    d = defaultdict(list)
    d['n_edges'] = n_edges_list

    model_name_noslash = model_name.split('/')[-1]
    model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device='cuda',
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    metric = get_metric(evaluation_metric_name, task=task, model=model)
    dataset = EAPDataset(task=task, model_name=model_name)
    dataloader = dataset.to_dataloader(batch_size=batch_size)

    baseline = evaluate_baseline(model, dataloader, partial(metric, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, dataloader, partial(metric, loss=False, mean=False), run_corrupted=True).mean().item()
    d['baseline'] = [baseline for _ in n_edges_list]
    d['corrupted_baseline'] = [corrupted_baseline for _ in n_edges_list]

    if ('kl' not in attribution_metric_name) and ('js' not in attribution_metric_name):
        graph = Graph.from_json(f'graphs/{model_name_noslash}/ig_test/{task}_{attribution_metric_name}_vanilla.json')
        for n_edges in n_edges_list:
            graph.apply_greedy(n_edges, absolute=True)
            graph.prune_dead_nodes()
            d[f'n_edges_vanilla'].append(graph.count_included_edges())
            results = evaluate_graph(model, graph, dataloader, partial(metric, loss=False, mean=False), quiet=True).mean().item()
            d[f'results_vanilla'].append(results)
    for n_iter in tqdm([2,3,5,7, 10, 15, 20, 30, 50]):
        graph = Graph.from_json(f'graphs/{model_name_noslash}/ig_test/{task}_{attribution_metric_name}_{n_iter}.json')
        for n_edges in n_edges_list:
            graph.apply_greedy(n_edges, absolute=True)
            graph.prune_dead_nodes()
            d[f'n_edges_{n_iter}'].append(graph.count_included_edges())
            results = evaluate_graph(model, graph, dataloader, partial(metric, loss=False, mean=False), quiet=True).mean().item()
            d[f'results_{n_iter}'].append(results)
    df = pd.DataFrame.from_dict(d)
    df.to_csv(f'results/ig_test/{model_name_noslash}/csv/{task}_{evaluation_metric_name}_ig_test.csv', index=False)

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--attribution_metric', type=str, required=True)
    parser.add_argument('--eval_metric', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--end', type=int, default=1001)

    args = parser.parse_args()
    model_name = args.model 
    task = args.task
    attribution_metric_name = args.attribution_metric
    evaluation_metric_name = args.eval_metric
    batch_size = args.batch_size

    attribute_graphs(model_name, task, attribution_metric_name, batch_size)
    evaluate_graphs(model_name, task, attribution_metric_name, evaluation_metric_name, batch_size, end=args.end)