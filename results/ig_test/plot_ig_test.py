#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
#%%
model = 'gpt2'
model_name_noslash = model.split('/')[-1]
#%%
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

#%%
task_names = ['ioi', 'greater-than', 'gender-bias']
#['ioi', 'greater-than', 'gender-bias', 'sva', 'fact-retrieval-comma', 'hypernymy-comma']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,3), sharey=True)
for i, (task_name, ax) in enumerate(zip(task_names, axs.flat)):
    metric = task_to_metric(task_name)
    df = pd.read_csv(f'{model_name_noslash}/csv/{task_name}_{metric}_ig_test.csv')
    if task_name == 'greater-than':
        df['corrupted_baseline'] = [-0.4565732181072235 for _ in range(len(df))]

    df[f'results_vanilla'] = (df[f'results_vanilla'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])

    ax.plot(df[f'n_edges_vanilla'].tolist(), df[f'results_vanilla'].tolist(), label=f'EAP')
    
    for n_iter in [2,3,5,7, 10, 15, 20, 30, 50]:
        df[f'results_{n_iter}'] = (df[f'results_{n_iter}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])

        ax.plot(df[f'n_edges_{n_iter}'].tolist(), df[f'results_{n_iter}'].tolist(), label=f'EAP-IG {n_iter} steps')

    
    
    if i == 0:
        ax.set_ylabel('Normalized faithfulness')

    ax.yaxis.set_tick_params(labelbottom=True)
    ax.set_title(f'{display_name(task_name)}')

handles, labels = ax.get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5)
xlabel = fig.text(0.5, -0.02, f'Edges included (/32491)', ha='center')

Path(f'{model_name_noslash}/png').mkdir(exist_ok=True, parents=True)
Path(f'{model_name_noslash}/pdf').mkdir(exist_ok=True, parents=True)
fig.savefig(f'{model_name_noslash}/png/ig_all.png', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.savefig(f'{model_name_noslash}/pdf/ig_all.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.show()
# %%
