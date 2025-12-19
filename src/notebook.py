#%%
from glob import glob
import polars as pl
import json
from os import path
from random import randint
def read_agnostic(d):
    txts = {
        path.basename(p): open(p, 'r').read().split('\n') for p in glob(f'{d}/*.txt')
    }

    parquet_paths = glob(f'{d}/*.parquet')
    parquets = {
        path.basename(p): pl.read_parquet(p) for p in parquet_paths
    }

    subdirs = glob(d+'/*')
    subdirs = [subd for subd in subdirs if path.isdir(subd)]

    dir_contents = {subd: {} for subd in subdirs}
    for subd in subdirs:
        train_id_paths = glob(subd+'/train_ids_*.txt')

        folds = []
        train_id_paths.sort()
        for p in train_id_paths:
            train_ids = open(p, 'r').read().split('\n')
            test_ids = open(p.replace('train_', 'test_'), 'r').read().split('\n')
            folds.append({'train_ids': train_ids, 'test_ids': test_ids})
        
        dir_contents[path.basename(subd)] = folds

    return {
        'lists': txts,
        'parquets': parquets,
        'fold_dirs': dir_contents,
        'params': json.load(open(d+'/params.json', 'r')),
    }

broken_dir = '/home/pita/fs/repos/mf_swarm/work/3e/33dcfcf997ed7e89e84be8bdb7629b/dataset'
vanilla_dir = '/home/pita/fs/experiments/mf_datasets/full_swarm_2025-12-18_22-03-53'

df_broken = read_agnostic(broken_dir)
df_vanilla = read_agnostic(vanilla_dir)

ids_n = len(df_vanilla['lists']['ids.txt'])
for _ in range(1000):
    index = randint(0, ids_n)
    id1 = df_vanilla['lists']['ids.txt'][index]
    id2 = df_broken['lists']['ids.txt'][index]
    assert id1 == id2

#%%

name_1_v = 'traintest_Level-1_Freq-3648-62610_N-4.parquet'
name_1_b = 'features_traintest.parquet'

ids_v = df_vanilla['parquets'][name_1_v]['id'].to_list()
ids_b = df_broken['parquets'][name_1_b]['id'].to_list()
n_intersect = len(set(ids_v).intersection(ids_b))
print('Intersection:', n_intersect)
assert n_intersect == len(ids_v)

uniprot_to_index = {}
for ids, name in [(ids_v, 'vanilla'), (ids_b, 'broken')]:
    for i, uniprot in enumerate(ids):
        if not uniprot in uniprot_to_index:
            uniprot_to_index[uniprot] = {}
        uniprot_to_index[uniprot][name] = i


#%%
from random import sample
from tqdm import tqdm

to_test = sample(ids_v, 9000)
b = df_broken['parquets'][name_1_b]
v = df_vanilla['parquets'][name_1_v]

for uniprot in tqdm(to_test):
    index_b = uniprot_to_index[uniprot]['broken']
    index_v = uniprot_to_index[uniprot]['vanilla']

    value_b = b[index_b]['taxa_256'][0].to_list()
    value_v = v[index_v]['taxa_256'][0].to_list()

    if value_b != value_v:
        print(value_b)
        print(value_v)
        print(uniprot)
        break
    
    for emb_name in ['ankh_base', 'esm2_t33']:
        value_b = b[index_b][emb_name][0]
        value_v = v[index_v][emb_name][0]
        ankh_b = value_b.to_list()
        ankh_v = value_v.to_list()
        diff = value_v - value_b
        diff = diff * diff
        total = (sum(ankh_b) + sum (ankh_v)) / 2
        diff_perc = sum(diff) / total
        if ankh_b != ankh_v:
            print(value_b)
            print(value_v)
            print(uniprot, diff_perc)
            break