#%% 
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
model = 'gpt2'
task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma']
logit_task_names = ['ioi', 'gender-bias', 'fact-retrieval-comma']
prob_task_names = ['greater-than', 'sva', 'hypernymy-comma']

with open(f'{model}/correlations.json', 'r') as f:
    correlations = json.load(f)
df = pd.read_csv(f'{model}/csv/all-overlap.csv')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,3), sharey=False)
ax1, ax2 = axs

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(task_names)]
color_cycle_evens = [color_cycle[2*i] for i in range(len(task_names)//2)]
color_cycle_odds = [color_cycle[2*i + 1] for i in range(len(task_names)//2)]

ind = np.arange(len(task_names))    # the x locations for the groups
evens = np.array([2*i for i in range (len(task_names)//2)])
odds = np.array([2*i + 1 for i in range (len(task_names)//2)])

width = 0.35         # the width of the bars
ax1.bar(evens, [correlations[f'{task}_vanilla_diff'] for task in logit_task_names], width, label='EAP', fill=False, edgecolor=color_cycle_evens, hatch='.')
ax1.bar(evens + width, [correlations[f'{task}_ig_diff'] for task in logit_task_names], width, label='EAP-IG', color=color_cycle_evens)

ax12 = ax1.twinx()

ax12.bar(odds, [correlations[f'{task}_vanilla_diff'] for task in prob_task_names], width, label='EAP', fill=False, edgecolor=color_cycle_odds, hatch='.')
ax12.bar(odds + width, [correlations[f'{task}_ig_diff'] for task in prob_task_names], width, label='EAP-IG', color=color_cycle_odds)


ax1.set_xticks(ind + width / 2, labels=[display_name(task) for task in task_names])
ax1.set_title("Activation patching score correlation by method / task")
ax1.set_ylabel("Pearson's r (correlation)")
ax1.set_xlabel("Task")
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ind = np.arange(len(task_names))    # the x locations for the groups
width = 0.35         # the width of the bars
ax2.bar(ind, [correlations[f'{task}_vanilla_kendall'] for task in task_names], width, label='EAP', fill=False, edgecolor=color_cycle, hatch='.')
ax2.bar(ind + width, [correlations[f'{task}_ig_kendall'] for task in task_names], width, label='EAP-IG', color=color_cycle)
ax2.set_xticks(ind + width / 2, labels=[display_name(task) for task in task_names])
ax2.set_title("Activation patching score correlation by method / task")
ax2.set_ylabel("Pearson's r (correlation)")
ax2.set_xlabel("Task")
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(12))#list(range(0, 12, 2)) + list(range(1,12,2))
lgd = fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=6,loc='lower center',bbox_to_anchor=(0.5, -0.14))

ax2.set_title("Average overlaps vs. top-n elements")# of top-n elements of activation \n patching edge rankings vs. EAP/EAP-IG edge rankings")
fig.tight_layout()
fig.savefig(f'{model}/png/all-overlap.png', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.savefig(f'{model}/pdf/all-overlap.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.show()
# %%
