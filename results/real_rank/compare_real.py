#%%
import json 
from collections import defaultdict

from scipy.stats import kendalltau, pearsonr
import pandas as pd
import torch
import matplotlib.pyplot as plt

from eap.graph import Graph
#%%
def agreement(x,y,d):
    xs = set()
    ys = set()
    intersection_sizes = []
    for i in range(d):
        xs.add(x[i])
        ys.add(y[i])
        intersection_sizes.append(len(xs & ys))
    intersection_sizes =  torch.tensor(intersection_sizes)
    ds = torch.arange(d) + 1
    agreements = intersection_sizes / ds 
    return agreements

def average_overlap(x,y,d):
    agreements = agreement(x,y,d)
    return agreements.mean()

def rank_biased_overlap(x,y,p):
    d = min(len(x), len(y))
    agreements = agreement(x,y,d)
    exps = torch.full((d,), p) ** torch.arange(d)
    return (1-p) * torch.dot(agreements, exps)
    
def intersection_size(x, y, d):
    return len(set(x[:d]) & set(y[:d]))

prob_diff_tasks = ['greater-than', 'fact-retrieval', 'sva']
def task_to_metric(task):
    if any(ptask in task for ptask in prob_diff_tasks):
        return 'prob_diff'
    return 'logit_diff'

display_name_dict = {'ioi': 'IOI', 'greater-than': "Greater-Than", 'gender-bias': 'Gender-Bias', 'sva': 'SVA', 'fact-retrieval': 'Country-Capital', 'hypernymy': 'Hypernymy'}
def display_name(task):
    if '-comma' in task:
        task = task[:-6]
    return display_name_dict[task] 
#%%
task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma']
k=50
model = 'gpt2'
overlaps = {'depth': list(range(1, 401))}
circuit_overlap_depth = list(range(25, 601))
circuit_node_overlaps = defaultdict(list)
circuit_edge_overlaps = defaultdict(list)
circuit_node_overlaps['depth'] = circuit_overlap_depth
circuit_edge_overlaps['depth'] = circuit_overlap_depth
correlations = {}
for i, task in enumerate(task_names):
    print()
    print(task)
    metric = task_to_metric(task)
    g_real = Graph.from_json(f'../../graphs/{model}/real_test/{task}_{metric}_real.json')
    g_vanilla = Graph.from_json(f'../../graphs/{model}/real_test/{task}_{metric}_vanilla.json')
    g_ig = Graph.from_json(f'../../graphs/{model}/real_test/{task}_{metric}_ig.json')

    for i in circuit_overlap_depth:
        nodes = {}
        edges = {}
        for circ, name in zip([g_real, g_vanilla, g_ig], ['real', 'EAP', 'EAP-IG']):
            circ.apply_greedy(i, absolute=True)
            circ.prune_dead_nodes()
            nodes[name] = {node.name for node in circ.nodes.values() if node.in_graph}
            edges[name] = {edge.name for edge in circ.edges.values() if edge.in_graph}

        for pairing in ['real_EAP', 'real_EAP-IG', 'EAP_EAP-IG']:
            n1, n2, = pairing.split('_')
            node_intersection = nodes[n1] & nodes[n2]
            node_union = nodes[n1] | nodes[n2]
            node_iou = len(node_intersection) / len(node_union) if node_union else 0
            circuit_node_overlaps[f'{task}_{pairing}'].append(node_iou)

            edge_intersection = edges[n1] & edges[n2]
            edge_union = edges[n1] | edges[n2]
            edge_iou = len(edge_intersection) / len(edge_union) if edge_union else 0
            circuit_edge_overlaps[f'{task}_{pairing}'].append(edge_iou)
        
    edge_order = list(g_real.edges.keys())
    real_scores = torch.tensor([g_real.edges[edge].score for edge in edge_order])
    vanilla_scores = torch.tensor([g_vanilla.edges[edge].score for edge in edge_order])
    ig_scores = torch.tensor([g_ig.edges[edge].score for edge in edge_order])

    real_order = torch.argsort(real_scores.abs(), descending=True)
    vanilla_order = torch.argsort(vanilla_scores.abs(), descending=True)
    ig_order = torch.argsort(ig_scores.abs(), descending=True)

    real_scores_real_order = real_scores.abs()[real_order]
    vanilla_scores_real_order = vanilla_scores.abs()[real_order]
    ig_scores_real_order = ig_scores.abs()[real_order]
    
    vanilla_score_diff = (real_scores - vanilla_scores).abs().mean().item()
    ig_score_diff = (real_scores - ig_scores).abs().mean().item()
    
    correlations[f'{task}_vanilla_diff'] = vanilla_score_diff
    correlations[f'{task}_ig_diff'] = ig_score_diff

    corr_vanilla = pearsonr(real_scores.abs(), vanilla_scores.abs())
    corr_ig = pearsonr(real_scores.abs(), ig_scores.abs())
    #print(corr_vanilla)
    #print(corr_ig)

    correlations[f'{task}_vanilla'] = corr_vanilla.statistic
    correlations[f'{task}_ig'] = corr_ig.statistic

    kendall_vanilla = kendalltau(real_scores.abs(), vanilla_scores.abs())
    kendall_ig = kendalltau(real_scores.abs(), ig_scores.abs())
    
    correlations[f'{task}_vanilla_kendall'] = kendall_vanilla.statistic
    correlations[f'{task}_ig_kendall'] = kendall_ig.statistic
    
    correlations[f'{task}_vanilla_kendall'] = kendall_vanilla.statistic
    correlations[f'{task}_ig_kendall'] = kendall_ig.statistic

    real_order_list = real_order.tolist()
    vanilla_order_list = vanilla_order.tolist()
    ig_order_list = ig_order.tolist()

    ds = list(range(1, 401))
    vanilla_averages = [average_overlap(real_order_list, vanilla_order_list, d).item() for d in ds]
    ig_averages = [average_overlap(real_order_list, ig_order_list, d).item() for d in ds]
    overlaps[f'{task}_vanilla'] = vanilla_averages
    overlaps[f'{task}_ig'] = ig_averages

with open(f'{model}/correlations.json', 'w') as f:
    json.dump(correlations, f)
#%%
df = pd.DataFrame.from_dict(overlaps)
df.to_csv(f'{model}/csv/all-overlap.csv', index=False)

node_df = pd.DataFrame.from_dict(circuit_node_overlaps)
node_df.to_csv(f'{model}/csv/node-overlap.csv', index=False)

edge_df = pd.DataFrame.from_dict(circuit_edge_overlaps)
edge_df.to_csv(f'{model}/csv/edge-overlap.csv', index=False)
#%%
fig, ax = plt.subplots()
for i, task in enumerate(task_names):
    line = ax.plot(node_df['depth'], node_df[f'{task}_real_EAP'], label=f'{task}-EAP')
    ax.plot(node_df['depth'], node_df[f'{task}_real_EAP-IG'], label=f'{task}-EAP-IG', color=line[0].get_color(), linestyle='dotted')

ax.set_ylabel('Average overlap')
handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(0, 12, 2)) + list(range(1,12,2))
lgd = fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=2,loc='lower right',bbox_to_anchor=(0.95, 0.15))#fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5)
xlabel = ax.set_xlabel('Top n') #fig.text(0.5, 0.00, f'Top n', ha='center')
ax.set_title("Average overlap of top-n elements of activation \n patching edge rankings vs. EAP/EAP-IG edge rankings")
fig.tight_layout()
#fig.savefig(f'{model}/png/all-overlap.png', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
#fig.savefig(f'{model}/pdf/all-overlap.pdf', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
fig.show()
#%%
fig, ax = plt.subplots()
for i, task in enumerate(task_names):
    line = ax.plot(edge_df['depth'], edge_df[f'{task}_real_EAP'], label=f'{task}-EAP')
    ax.plot(edge_df['depth'], edge_df[f'{task}_real_EAP-IG'], label=f'{task}-EAP-IG', color=line[0].get_color(), linestyle='dotted')

ax.set_ylabel('Average overlap')
handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(0, 12, 2)) + list(range(1,12,2))
lgd = fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=2,loc='lower right',bbox_to_anchor=(0.95, 0.15))#fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5)
xlabel = ax.set_xlabel('Top n') #fig.text(0.5, 0.00, f'Top n', ha='center')
ax.set_title("Average overlap of top-n elements of activation \n patching edge rankings vs. EAP/EAP-IG edge rankings")
fig.tight_layout()
#fig.savefig(f'{model}/png/all-overlap.png', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
#fig.savefig(f'{model}/pdf/all-overlap.pdf', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
fig.show()
#%%
fig, ax = plt.subplots()
for i, task in enumerate(task_names):
    line = ax.plot(df['depth'], df[f'{task}_vanilla'], label=f'{task}-EAP')
    ax.plot(df['depth'], df[f'{task}_ig'], label=f'{task}-EAP-IG', color=line[0].get_color(), linestyle='dotted')

ax.set_ylabel('Average overlap')
handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(0, 12, 2)) + list(range(1,12,2))
lgd = fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=2,loc='lower right',bbox_to_anchor=(0.95, 0.15))#fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5)
xlabel = ax.set_xlabel('Top n') #fig.text(0.5, 0.00, f'Top n', ha='center')
ax.set_title("Average overlap of top-n elements of activation \n patching edge rankings vs. EAP/EAP-IG edge rankings")
fig.tight_layout()
#fig.savefig(f'{model}/png/all-overlap.png', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
#fig.savefig(f'{model}/pdf/all-overlap.pdf', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
fig.show()
# %%
