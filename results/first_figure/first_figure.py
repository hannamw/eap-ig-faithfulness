#%%
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
#%%
fig, axs = plt.subplots(ncols=2, figsize=(10,2.5),)
ax1, ax2 = axs
xs = torch.linspace(-4, 2, 100)
ys = F.gelu(xs).numpy()
xs = xs.numpy()
ax1.plot(xs, ys, color='black')

points_x = torch.linspace(-3.5, 1.5, 6)
points_y = F.gelu(points_x).numpy()
points_x = points_x.numpy()
ax1.scatter(points_x, points_y, color=['green'] + ['blue']* (len(points_x) - 2) + ['red'])
ax1.set_xlabel('Edge Activation')
ax1.set_ylabel('Loss')
ax1.set_title('Integrated Gradients on GELU')
ax1.annotate("z", [points_x[0] - 0.2, points_y[0] + 0.2], fontsize=14, fontweight='bold')
ax1.annotate('"Rome, the capital of"', [points_x[0] - 0.6, points_y[0] + 0.6], fontsize=14,)

ax1.annotate("z'", [points_x[-1] - 0.4, points_y[-1] - 0.1], fontsize=14, fontweight='bold')
ax1.annotate('"Paris, the capital of"', [points_x[-1] - 3.0, points_y[-1] + 0.3], fontsize=14)

display_name_dict = {'ioi': 'IOI', 'greater-than': "Greater-Than", 'greater-than-price': "Greater-Than (Price)", 'greater-than-sequence': "Greater-Than (Sequence)", 'gender-bias': 'Gender-Bias', 'sva': 'SVA', 'fact-retrieval': 'Country-Capital', 'hypernymy': 'Hypernymy'}
def display_name(task):
    if '-comma' in task:
        task = task[:-6]
    return display_name_dict[task] 

prob_diff_tasks = ['greater-than', 'fact-retrieval', 'sva']
def task_to_metric(task):
    if any(ptask in task for ptask in prob_diff_tasks):
        return 'prob_diff'
    return 'logit_diff'


model = 'gpt2'
task_name = 'fact-retrieval-comma'
df = pd.read_csv(f'../pareto/{model}/csv/{task_name}.csv')
for losstype in ['EAP', 'EAP-IG', 'EAP-IG-KL', 'EAP-IG-JS']:
    df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])

ax2.plot(df['edges_EAP'], df['loss_EAP'], label='EAP')
ax2.plot(df['edges_EAP-IG'], df['loss_EAP-IG'], label='EAP-IG')

ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax2.set_title(f'Circuit Faithfulness on the {display_name(task_name)} Task')

handles, labels = ax2.get_legend_handles_labels()
lgd = ax2.legend(handles, labels, loc='lower right')
xlabel = ax2.set_xlabel(f'Edges included (/32491)')
ylabel = ax2.set_ylabel('Normalized faithfulness')
fig.tight_layout()
fig.savefig(f'first_figure_more_text.png', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
fig.savefig(f'first_figure_more_text.pdf', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
fig.show()
# %%
