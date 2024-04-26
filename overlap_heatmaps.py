#%%
from collections import Counter

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import hypergeom

from eap.graph import Graph
# %%

model_name = 'gpt2'
model_name_noslash = model_name.split('/')[-1]

display_name_dict = {'ioi': 'IOI', 'greater-than': "Greater-Than (GT)", 'greater-than-price': "GT (Price)", 'greater-than-sequence': "GT (Sequence)", 'gender-bias': 'Gender-Bias', 'sva': 'SVA', 'fact-retrieval': 'Country-Capital', 'hypernymy': 'Hypernymy'}
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
def make_graph_hm(z, hovertext, graphs_names, title):
    heat = go.Heatmap(z=z[1:, :-1],
                  x=graphs_names[:-1],
                  y=graphs_names[1:],
                  colorbar={"title": 'IoU'},
                  xgap=1, ygap=1,
                  colorscale='Blues',
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                  hovertext=hovertext,
                  #hoverinfo='text'
                   )
    layout = go.Layout(title_text=title, title_x=0.5, 
                    width=600, height=600,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed')
    
    fig=go.Figure(data=[heat], layout=layout)        
    return fig

def graph_analysis(g1, g2):
    edges_g1 = {edge.name for edge in g1.edges.values() if edge.in_graph}
    edges_g2 = {edge.name for edge in g2.edges.values() if edge.in_graph}
    edge_intersection = edges_g1 & edges_g2 
    edge_union = edges_g1 | edges_g2
    #print(len(edge_intersection), len(edge_union), len(edge_intersection) / len(edge_union))
    x = len(edge_intersection)
    M  = len(g1.edges) 
    n = len(edges_g1)
    N = len(edges_g2)
    iou_edge = len(edge_intersection) / len(edge_union)
    p_edge = 1 - hypergeom.cdf(x, M, n, N)
    #print(1 - hypergeom.cdf(x, M, N, n))

    nodes_g1 = {node.name for node in g1.nodes.values() if node.in_graph} - {'inputs', 'logits'}
    nodes_g2 = {node.name for node in g2.nodes.values() if node.in_graph} - {'inputs', 'logits'}
    node_intersection = nodes_g1 & nodes_g2 
    node_union = nodes_g1 | nodes_g2
    #print(len(node_intersection), len(node_union), len(node_intersection) / len(node_union))
    x = len(node_intersection)
    M  = len(g1.nodes) - 2
    n = len(nodes_g1)
    N = len(nodes_g2)
    p_node = 1 - hypergeom.cdf(x, M, n, N)
    iou_node = len(node_intersection) / len(node_union)
    #print(1 - hypergeom.cdf(x, M, N, n))

    # directional measures:
    edge_overlap = len(edge_intersection) / len(edges_g1)
    node_overlap = len(node_intersection) / len(nodes_g1)
    return p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap

def make_comparison_heatmap(graphs: Graph, graphs_names, edge_thresholds, title):
    edge_counts = [[] for _ in graphs]
    node_counts = [[] for _ in graphs]
    for graph, n_edges in zip(graphs, edge_thresholds):
        graph.apply_greedy(n_edges, absolute=True)
        graph.prune_dead_nodes()

    pes = np.zeros((len(graphs), len(graphs)))
    ies = np.zeros((len(graphs), len(graphs)))
    pns = np.zeros((len(graphs), len(graphs)))
    ins = np.zeros((len(graphs), len(graphs)))
    eos = np.zeros((len(graphs), len(graphs)))
    nos = np.zeros((len(graphs), len(graphs)))
    
    for i, (g1, n1) in enumerate(zip(graphs, graphs_names)):
        edge_counts[i].append(g1.count_included_edges())
        node_counts[i].append(g1.count_included_nodes())
        for j, (g2, n2) in enumerate(zip(graphs, graphs_names)):
            p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap = graph_analysis(g1,g2)
            pes[i,j] = p_edge
            ies[i,j] = iou_edge
            pns[i,j] = p_node
            ins[i,j] = iou_node
            eos[i,j] = edge_overlap 
            nos[i,j] = node_overlap

    display_names = [display_name(name) for name in graphs_names]

    edge_iou_df = pd.DataFrame.from_dict({gn: ies[:, i] for i, gn in enumerate(graphs_names)})
    edge_iou_p_df = pd.DataFrame.from_dict({gn: pes[:, i] for i, gn in enumerate(graphs_names)})

    edge_iou_df.to_csv('results/jaccard/csv/edge_ious.csv', index=False)
    edge_iou_p_df.to_csv('results/jaccard/csv/edge_iou_ps.csv', index=False)

    ius = np.triu_indices(len(graphs))
    ies[ius] = np.nan
    pes[ius] = np.nan

    fig = make_graph_hm(ies, pes, display_names, f'Edge Intersection over Union (85%)')
    fig.write_image(f'results/jaccard/png/{title}_edges.png')
    fig.write_image(f'results/jaccard/pdf/{title}_edges.pdf')
    
    
    fig = make_graph_hm(pes, ies, display_names, f'Edge p-value (85%)')
    fig.write_image(f'results/jaccard/png/{title}_edges_p.png')
    fig.write_image(f'results/jaccard/pdf/{title}_edges_p.pdf')


    node_iou_df = pd.DataFrame.from_dict({gn: ins[:, i] for i, gn in enumerate(graphs_names)})
    node_iou_p_df = pd.DataFrame.from_dict({gn: pns[:, i] for i, gn in enumerate(graphs_names)})

    node_iou_df.to_csv('results/jaccard/csv/node_ious.csv', index=False)
    node_iou_p_df.to_csv('results/jaccard/csv/node_iou_ps.csv', index=False)

    ius = np.triu_indices(len(graphs))
    ins[ius] = np.nan
    pns[ius] = np.nan

    fig = make_graph_hm(ins, pns, display_names, f'Node Intersection over Union (85%)')
    fig.write_image(f'results/jaccard/png/{title}_nodes.png')
    fig.write_image(f'results/jaccard/pdf/{title}_nodes.pdf')

    fig = make_graph_hm(pns, ins, display_names, f'Node p-value (85%)')
    fig.write_image(f'results/jaccard/png/{title}_nodes_p.png')
    fig.write_image(f'results/jaccard/pdf/{title}_nodes_p.pdf')

    edge_overlap_df = pd.DataFrame.from_dict({gn: eos[:, i] for i, gn in enumerate(graphs_names)})
    node_overlap_df = pd.DataFrame.from_dict({gn: nos[:, i] for i, gn in enumerate(graphs_names)})

    edge_overlap_df.to_csv('results/jaccard/csv/edge_overlap.csv', index=False)
    node_overlap_df.to_csv('results/jaccard/csv/node_overlap.csv', index=False)

    fig = make_graph_hm(eos, pes, display_names, f'Edge Overlap')
    fig.write_image(f'results/jaccard/png/{title}_edge_overlap.png')
    fig.write_image(f'results/jaccard/pdf/{title}_edge_overlap.pdf')

    fig = make_graph_hm(nos, ins, display_names, f'Node Overlap')
    fig.write_image(f'results/jaccard/png/{title}_node_overlap.png')
    fig.write_image(f'results/jaccard/pdf/{title}_node_overlap.pdf')

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

make_comparison_heatmap(graphs, graphs_names, edge_thresholds, "real_task_overlap")
