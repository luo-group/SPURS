import pandas as pd
import subprocess
import os


df = pd.read_csv('../data/fitness/DMS_substitutions.csv')

filtered_df = df[df['coarse_selection_type'].isin(['Binding', 'Activity', 'Expression','OrganismalFitness'])]

for j in [48,96,144,192,-1]:
    for i, row in filtered_df.iterrows():
        if i >3:
            break
        
        cmd = f"python src/evaluate.py {row.DMS_id} core --n_seeds=20 --n_threads=20 --n_train={j}"
        subprocess.run(cmd, shell=True)
    

