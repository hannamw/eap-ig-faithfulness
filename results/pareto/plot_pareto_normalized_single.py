#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
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

#%%
model = 'gpt2'
for task_name in ['greater-than-sequence', 'greater-than-price']:
    fig, ax = plt.subplots()
    df = pd.read_csv(f'{model}/csv/{task_name}.csv')
    for losstype in ['EAP', 'EAP-IG', 'EAP-IG-KL']:#, 'EAP-IG-JS']:
        df[f'loss_{losstype}'] = (df[f'loss_{losstype}'] - df['corrupted_baseline']) /(df['baseline'] - df['corrupted_baseline'])

    ax.plot(df['edges_EAP'], df['loss_EAP'], label='EAP')
    ax.plot(df['edges_EAP-IG'], df['loss_EAP-IG'], label='EAP-IG')
    ax.plot(df['edges_EAP-IG-KL'], df['loss_EAP-IG-KL'], label='EAP-IG-KL')
    #ax.plot(df['edges_EAP-IG-JS'], df['loss_EAP-IG-JS'], label='EAP-IG-JS')

    ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    ax.set_title(f'{display_name(task_name)}')
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=12)
    xlabel = fig.text(0.5, 0.00, f'Edges included (/32491)', ha='center', fontsize=12)
    ylabel = fig.text(-0.00, 0.5, 'Normalized faithfulness', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{model}/png/{task_name}.png', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
    fig.savefig(f'{model}/pdf/{task_name}.pdf', bbox_extra_artists=[lgd, xlabel, ylabel], bbox_inches='tight')
    fig.show()
