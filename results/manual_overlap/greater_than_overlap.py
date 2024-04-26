#%%
from functools import partial
from pathlib import Path 
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eap.graph import Graph
#%%
g0 = Graph.from_json('../../graphs/gpt2/greater-than_vanilla.json')
g1 = Graph.from_json('../../graphs/gpt2/greater-than_task.json')
g2 = Graph.from_json('../../graphs/gpt2/greater-than_task.json')
for edge in g2.edges.values():
    edge.in_graph=False

for node in g2.nodes.values():
    node.in_graph=False

input_node = g2.nodes['input']
low_attn = ['a0.h3', 'a0.h5']
low_mlps = ['a0.h1', 'm0', 'm1', 'm2', 'm3']
mid_attn = ['a5.h1', 'a5.h5','a6.h1','a6.h9', 'a7.h10', 'a8.h8', 'a8.h11', 'a9.h1']
high_mlps = [f'm{i}' for i in range(8,12)]
logit_node = g2.nodes['logits']

input_node.in_graph = True
logit_node.in_graph = True 
for node_name in low_attn + low_mlps + mid_attn + high_mlps:
    g2.nodes[node_name].in_graph = True

for node in low_attn + low_mlps:
    if f'input->{node}' in g2.edges:
        g2.edges[f'input->{node}'].in_graph = True 
    else: 
        for letter in 'qkv':
            g2.edges[f'input->{node}<{letter}>'].in_graph = True 

    for node2 in mid_attn:
        for letter in 'qkv':
            g2.edges[f'{node}->{node2}<{letter}>'].in_graph = True

for i, node in enumerate(low_mlps):
    for node2 in low_mlps[i+1:]:
        g2.edges[f'{node}->{node2}'].in_graph = True

for node in mid_attn:
    g2.edges[f'{node}->logits'].in_graph=True
    for node2 in high_mlps:
        edge_str = f'{node}->{node2}' 
        if edge_str in g2.edges:
            g2.edges[edge_str].in_graph=True

for i, node in enumerate(high_mlps):
    g2.edges[f'{node}->logits'].in_graph=True
    for node2 in high_mlps[i+1:]:
        g2.edges[f'{node}->{node2}'].in_graph = True
#%%
edge_counts = list(range(1, 200))
d = defaultdict(list)
d['edges']  = edge_counts
ref_nodes = {node.name for node in g2.nodes.values() if node.in_graph} - {'logits', 'input'}
ref_edges = {edge.name for edge in g2.edges.values() if edge.in_graph}
for i in edge_counts:
    g0.apply_topn(i, absolute=True)
    g1.apply_topn(i, absolute=True)

    g0_nodes = {node.name for node in g0.nodes.values() if node.in_graph}- {'logits', 'input'}
    g0_edges = {edge.name for edge in g0.edges.values() if edge.in_graph}
    g1_nodes = {node.name for node in g1.nodes.values() if node.in_graph}- {'logits', 'input'}
    g1_edges = {edge.name for edge in g1.edges.values() if edge.in_graph}

    d['EAP_node_precision'].append(len(g0_nodes & ref_nodes) / len(g0_nodes))
    d['EAP_node_recall'].append(len(g0_nodes & ref_nodes) / len(ref_nodes))

    d['EAP_edge_precision'].append(len(g0_edges & ref_edges) / len(g0_edges))
    d['EAP_edge_recall'].append(len(g0_edges & ref_edges) / len(ref_edges))

    d['EAP-IG_node_precision'].append(len(g1_nodes & ref_nodes) / len(g1_nodes))
    d['EAP-IG_node_recall'].append(len(g1_nodes & ref_nodes) / len(ref_nodes))

    d['EAP-IG_edge_precision'].append(len(g1_edges & ref_edges) / len(g1_edges))
    d['EAP-IG_edge_recall'].append(len(g1_edges & ref_edges) / len(ref_edges))
# %%
df = pd.DataFrame.from_dict(d)
df.to_csv('csv/greater-than_node_edge_precision_recall.csv',index=False)
#%%
fig, ax = plt.subplots()
ax.plot(df['EAP_node_recall'], df['EAP_node_precision'], label='EAP (node)', c = 'orange', linestyle ='dotted')
ax.plot(df['EAP-IG_node_recall'], df['EAP-IG_node_precision'], label = 'EAP-IG  (node)', c='orange')

ax.plot(df['EAP_edge_recall'], df['EAP_edge_precision'], label='EAP (edge)', c = 'blue', linestyle ='dotted')
ax.plot(df['EAP-IG_edge_recall'], df['EAP-IG_edge_precision'], label = 'EAP-IG  (edge)', c='blue')

ax.set_ylim(0.45, 1.03)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title("EAP Node and Edge Precision and Recall on Greater-Than")
lgd = ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.25), ncol=4)
fig.tight_layout()
fig.show()
fig.savefig('png/greater-than-PR.png', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.savefig('pdf/greater-than-PR.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
# %%
