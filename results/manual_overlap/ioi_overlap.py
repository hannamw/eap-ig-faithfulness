#%%
from functools import partial
from pathlib import Path 
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eap.graph import Graph, MLPNode
#%%
# Instantiate a graph with a model
g0 = Graph.from_json('../../graphs/gpt2/ioi_vanilla.json')
g1 = Graph.from_json('../../graphs/gpt2/ioi_task.json')
g2 = Graph.from_json('../../graphs/gpt2/ioi_task.json')
for edge in g2.edges.values():
    edge.score = 0.1
    edge.in_graph = False

circ = {
"name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        (4, 11),
    ],
}

for node in g2.nodes.values():
    node.in_graph = False

for nodelist in circ.values():
    for l,h in nodelist:
        g2.nodes[f'a{l}.h{h}'].in_graph = True

for (l,h) in circ['previous token']:
    g2.edges[f'input->a{l}.h{h}<q>'].in_graph = True
    g2.edges[f'input->a{l}.h{h}<k>'].in_graph = True
    g2.edges[f'input->a{l}.h{h}<v>'].in_graph = True

    for (l2, h2) in circ['induction']:
        g2.edges[f'a{l}.h{h}->a{l2}.h{h2}<q>'].in_graph = True

for (l,h) in circ['duplicate token']:
    g2.edges[f'input->a{l}.h{h}<q>'].in_graph = True
    g2.edges[f'input->a{l}.h{h}<k>'].in_graph = True
    g2.edges[f'input->a{l}.h{h}<v>'].in_graph = True

    for (l2, h2) in circ['induction']:
        g2.edges[f'a{l}.h{h}->a{l2}.h{h2}<k>'].in_graph = True
        g2.edges[f'a{l}.h{h}->a{l2}.h{h2}<v>'].in_graph = True

for (l, h) in [(5, 5), (5, 8), (5, 9)]:
    g2.edges[f'a{l}.h{h}->a6.h9<q>'].in_graph = True

for (l, h) in [(10, 0),(10, 10),(10, 6),(10, 2),(10, 1),(11, 2), (11, 9),]:
    g2.edges[f'a9.h6->a{l}.h{h}<q>'].in_graph=True
    g2.edges[f'a9.h9->a{l}.h{h}<q>'].in_graph=True
    g2.edges[f'a9.h0->a{l}.h{h}<q>'].in_graph=True
    g2.edges[f'a9.h7->a{l}.h{h}<q>'].in_graph=True

for (l,h) in circ['s2 inhibition']:
    g2.edges[f'input->a{l}.h{h}<q>'].in_graph = True

    g2.edges[f'input->a{l}.h{h}<k>'].in_graph = True

    g2.edges[f'input->a{l}.h{h}<v>'].in_graph = True
    for (l2, h2) in circ['induction']:
        g2.edges[f'a{l2}.h{h2}->a{l}.h{h}<k>'].in_graph = True
        g2.edges[f'a{l2}.h{h2}->a{l}.h{h}<v>'].in_graph = True
    for (l2,h2) in circ['name mover'] + circ['negative']:
        g2.edges[f'a{l}.h{h}->a{l2}.h{h2}<q>'].in_graph = True

for (l,h) in circ['name mover']:
    g2.edges[f'input->a{l}.h{h}<k>'].in_graph = True
    g2.edges[f'input->a{l}.h{h}<v>'].in_graph = True

    g2.edges[f'a{l}.h{h}->logits'].in_graph = True

for (l,h) in circ['negative']:
    g2.edges[f'input->a{l}.h{h}<k>'].in_graph = True
    g2.edges[f'input->a{l}.h{h}<v>'].in_graph = True

    g2.edges[f'a{l}.h{h}->logits'].in_graph = True

g2.edges[f'a9.h7->m11'].in_graph = True
g2.edges[f'm11->logits'].in_graph = True

#%%
edge_counts = list(range(1, 200))
d = defaultdict(list)
d['edges']  = edge_counts
ref_nodes = {node.name for node in g2.nodes.values() if node.in_graph} - {'logits', 'input'}
ref_edges = {edge.name for edge in g2.edges.values() if edge.in_graph}
for i in edge_counts:
    g0.apply_topn(i, absolute=True)
    g1.apply_topn(i, absolute=True)

    g0_nodes = {node.name for node in g0.nodes.values() if node.in_graph and not isinstance(node, MLPNode)}- {'logits', 'input'}
    g0_edges = {edge.name for edge in g0.edges.values() if edge.in_graph and not 'm' in edge.name}
    g1_nodes = {node.name for node in g1.nodes.values() if node.in_graph and not isinstance(node, MLPNode)}- {'logits', 'input'}
    g1_edges = {edge.name for edge in g1.edges.values() if edge.in_graph and not 'm' in edge.name}

    d['EAP_node_precision'].append(len(g0_nodes & ref_nodes) / (len(g0_nodes) if len(g0_nodes) else 1))
    d['EAP_node_recall'].append(len(g0_nodes & ref_nodes) / len(ref_nodes))

    d['EAP_edge_precision'].append(len(g0_edges & ref_edges) / (len(g0_edges) if len(g0_edges) else 1))
    d['EAP_edge_recall'].append(len(g0_edges & ref_edges) / len(ref_edges))

    d['EAP-IG_node_precision'].append(len(g1_nodes & ref_nodes) / (len(g1_nodes) if len(g1_nodes) else 1))
    d['EAP-IG_node_recall'].append(len(g1_nodes & ref_nodes) / len(ref_nodes))

    d['EAP-IG_edge_precision'].append(len(g1_edges & ref_edges) / (len(g1_edges) if len(g1_edges) else 1))
    d['EAP-IG_edge_recall'].append(len(g1_edges & ref_edges) / len(ref_edges))
# %%
df = pd.DataFrame.from_dict(d)
df.to_csv('csv/ioi_node_edge_precision_recall.csv',index=False)
#%%
fig, ax = plt.subplots()
ax.plot(df['EAP_node_recall'], df['EAP_node_precision'], label='EAP (node)', c = 'orange', linestyle ='dotted')
ax.plot(df['EAP-IG_node_recall'], df['EAP-IG_node_precision'], label = 'EAP-IG  (node)', c='orange')

ax.plot(df['EAP_edge_recall'], df['EAP_edge_precision'], label='EAP (edge)', c = 'blue', linestyle ='dotted')
ax.plot(df['EAP-IG_edge_recall'], df['EAP-IG_edge_precision'], label = 'EAP-IG  (edge)', c='blue')

ax.set_ylim(0.53, 1.03)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title("EAP Node and Edge Precision and Recall on IOI")
lgd = ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.25), ncol=4)
fig.tight_layout()
fig.show()
fig.savefig('png/ioi-PR.png', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.savefig('pdf/ioi-PR.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
# %%
