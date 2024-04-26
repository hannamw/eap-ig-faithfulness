#%%
import pandas as pd
import matplotlib.pyplot as plt
display_name_dict = {'ioi': 'IOI', 'greater-than': "Greater-Than", 'gender-bias': 'Gender-Bias', 'sva': 'SVA', 'fact-retrieval': 'Country-Capital', 'hypernymy': 'Hypernymy'}
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

#%%
model = 'gpt2'
task_names = ['ioi', 'greater-than', 'sva', 'gender-bias', 'fact-retrieval-comma', 'hypernymy-comma']
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,5), sharey=True)
for i, (task_name, ax) in enumerate(zip(task_names, axs.flat)):
    df = pd.read_csv(f'../pareto/{model}/csv/{task_name}.csv')
    for losstype in ['EAP', 'EAP-IG', 'EAP-IG-KL']:
        df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])
        ax.plot(df[f'edges_{losstype}'], df[f'loss_{losstype}'], label=losstype, alpha=0.8)
    
    df = pd.read_csv(f'../real/{model}/csv/{task_name}.csv')
    for losstype in ['real', 'real-KL']:
        df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])
        ax.plot(df[f'edges_{losstype}'], df[f'loss_{losstype}'], label=metric_to_name(losstype), alpha=0.8) 
    
    ax.yaxis.set_tick_params(labelbottom=True)

    title_y = 0.88 #1.0 if i < 3 else 0.88
    ax.set_title(f'{display_name(task_name)}', y=title_y)

    if task_name == 'greater-than':
        # inset axes....
        x1, x2, y1, y2 = 0.0, 300.0, 0.6, 1.0  # subregion of the original image
        axins = ax.inset_axes(
            [0.4, 0.06, 0.57, 0.57],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        
        df = pd.read_csv(f'../pareto/{model}/csv/{task_name}.csv')
        for losstype in ['EAP', 'EAP-IG', 'EAP-IG-KL']:
            df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])
            axins.plot(df[f'edges_{losstype}'], df[f'loss_{losstype}'], label=losstype, alpha=0.8)
        
        df = pd.read_csv(f'../real/{model}/csv/{task_name}.csv')
        for losstype in ['real', 'real-KL']: #, 'EAP', 'EAP-IG', 'EAP-IG-KL']:
            df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])
            axins.plot(df[f'edges_{losstype}'], df[f'loss_{losstype}'], label=losstype, alpha=0.8) 

        ax.indicate_inset_zoom(axins, edgecolor="black")

handles, labels = ax.get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=12)
xlabel = fig.text(0.5, -0.015, f'Edges included (/32491)', ha='center', fontsize=12)
ylabel = fig.text(-0.00, 0.5, 'Normalized faithfulness', va='center', rotation='vertical', fontsize=12)
fig.tight_layout()
fig.savefig(f'{model}/png/real-pareto.png', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
fig.savefig(f'{model}/pdf/real-pareto.pdf', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
fig.show()
# %%
