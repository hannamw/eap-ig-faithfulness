#%%
import pandas as pd
import matplotlib.pyplot as plt 

#%%
model = 'gpt2'
node_df = pd.read_csv(f'{model}/csv/node-overlap.csv')
edge_df = pd.read_csv(f'{model}/csv/edge-overlap.csv')
task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma']

#%%
fig, axs = plt.subplots(ncols=2, sharey=True,figsize=(10,4))
ax1, ax2 = axs
for i, task in enumerate(task_names):
    line = ax1.plot(node_df['depth'], node_df[f'{task}_real_EAP'], label=f'{task}-EAP')
    ax1.plot(node_df['depth'], node_df[f'{task}_real_EAP-IG'], label=f'{task}-EAP-IG', color=line[0].get_color(), linestyle='dotted')

ax1.set_ylabel('Node IoU')

ax1.set_title("Node IoU of activation patching circuits vs.\n EAP/EAP-IG circuits at various n edges")

for i, task in enumerate(task_names):
    line = ax2.plot(edge_df['depth'], edge_df[f'{task}_real_EAP'], label=f'{task}-EAP', linestyle='dotted')
    ax2.plot(edge_df['depth'], edge_df[f'{task}_real_EAP-IG'], label=f'{task}-EAP-IG', color=line[0].get_color())

ax2.set_ylabel('Edge IoU')

xlabel = fig.text(0.5, 0.00, f'Edges included (/32491)', ha='center')
ax2.set_title("Edge IoU of activation patching circuits vs.\n EAP/EAP-IG circuits at various n edges")

handles, labels = plt.gca().get_legend_handles_labels()

order = list(range(12))
lgd = fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=6,loc='lower center',bbox_to_anchor=(0.5, -0.24))

fig.tight_layout()
fig.savefig(f'{model}/png/node-edge-overlap.png', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
fig.savefig(f'{model}/pdf/node-edge-overlap.pdf', bbox_extra_artists=[lgd, xlabel], bbox_inches='tight')
fig.show()
# %%
