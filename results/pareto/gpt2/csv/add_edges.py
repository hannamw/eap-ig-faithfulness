#%%
import pandas as pd
# %%
task_name = 'hypernymy-comma'
df = pd.read_csv(f'{task_name}.csv')
print(len(df))
# %%
df['edges'] = list(range(100, 2000, 100))
# %%
df.to_csv(f'{task_name}.csv', index=False)

# %%
