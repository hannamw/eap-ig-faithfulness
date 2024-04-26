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

def n_edges(model):
    if model == 'gpt2':
        return 32491
    elif model == 'gpt2-xl':
        return 2235025
    elif model =='pythia-2.8B':
        return 1591857
    else:
        raise ValueError(f'Bad model: {model}')

#%%
model = 'pythia-2.8B'
#task_names = ['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma']
task_names = ['ioi', 'greater-than', 'sva', 'gender-bias', 'fact-retrieval-comma', 'hypernymy-comma']
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6), sharey=True)
for task_name, ax in zip(task_names, axs.flat):
    df = pd.read_csv(f'{model}/csv/{task_name}.csv')
    for losstype in ['EAP', 'EAP-IG', 'EAP-IG-KL']:#, 'EAP-IG-JS']:
        df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])
    
    ax.plot(df['edges_EAP'], df['loss_EAP'], label='EAP')
    ax.plot(df['edges_EAP-IG'], df['loss_EAP-IG'], label='EAP-IG')
    ax.plot(df['edges_EAP-IG-KL'], df['loss_EAP-IG-KL'], label='EAP-IG-KL')
    #ax.plot(df['edges_EAP-IG-JS'], df['loss_EAP-IG-JS'], label='EAP-IG-JS')
    
    ax.yaxis.set_tick_params(labelbottom=True)
    ax.set_title(f'{display_name(task_name)}')
handles, labels = ax.get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=12)
xlabel = fig.text(0.5, -0.015, f'Edges included (/{n_edges(model)})', ha='center', fontsize=12)
ylabel = fig.text(-0.00, 0.5, 'Normalized faithfulness', va='center', rotation='vertical', fontsize=12)
fig.tight_layout()
fig.savefig(f'{model}/png/eap-ig-comparison.png', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
fig.savefig(f'{model}/pdf/eap-ig-comparison.pdf', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
fig.show()
# %%
