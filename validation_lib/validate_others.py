import gzip
import json
import polars as pl
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
import os

from stats_custom import calc_metrics_at_freq_threshold

def calc_metrics_for_tool(df_path, experimental_annots, max_n=3000):
    print(df_path)
    print('Loading')
    labels_path = df_path.replace('-preds.parquet', '-label_names.txt')
    labels_sequence = open(labels_path, 'r').read().split('\n')
    df = pl.read_parquet(df_path)
    ids_list = df['id'].to_list()
    true_labels = []
    
    print('Creating true labels')
    for uniprot in ids_list:
        true_vec = [1.0 if go in experimental_annots[uniprot] else 0.0 for go in labels_sequence]
        true_labels.append(true_vec)
    val_y_pred = df['labels'].to_list()
    if len(true_labels) > max_n:
        true_labels = np.asarray(true_labels)[:max_n]
        val_y_pred = np.asarray([np.array(v) for v in val_y_pred])[:max_n]
    else:
        true_labels = np.asarray(true_labels)
        val_y_pred = np.asarray([np.array(v) for v in val_y_pred])
    print(val_y_pred)
    col_sums = true_labels.sum(axis=0)
    #freq_thresholds = [9, 7, 5, 3, 1, 0]
    freq_thresholds = [5, 3, 0]
    for min_freq in freq_thresholds:
        print('min freq:', min_freq)
        print(f"Removed {np.sum(~(col_sums > min_freq))} columns with low frequency in true_labels.")
    metrics_dict = {}
    
    for min_freq in freq_thresholds:
        new_m = calc_metrics_at_freq_threshold(col_sums, true_labels, val_y_pred, 
            min_freq, labels_sequence, labels_path)
        metrics_dict[f'Min_Freq_{min_freq+1}'] = new_m
        print(json.dumps(new_m, indent=2))
    print(metrics_dict)
    return metrics_dict

#sys.argv = ['others/validate_others.py', '~/data/protein_dimension_db/release_1/', 
#    '40', '0.15', '~/data/mf_swarm_datasets/other_tools/']
prot_dimension_db_release_path = os.path.expanduser(sys.argv[1])
min_protein_annots = sys.argv[2]
val_part = sys.argv[3]
others_dir = os.path.expanduser(sys.argv[4])

print('Loading validation proteins')
validation_split_dir = prot_dimension_db_release_path + f'/validation_splits/min_prot{min_protein_annots}_val{val_part}'
val_ids = validation_split_dir+'/validation_ids.txt'
validation_proteins = open(val_ids, 'r').read().split('\n')

print('Loading experimental annotations')
mf_annotations_path = f'{prot_dimension_db_release_path}/go.experimental.mf.tsv.gz'
annots_cache = f'{validation_split_dir}/experimental_annots.json'
if os.path.exists(annots_cache):
    experimental_annots = json.load(open(annots_cache, 'r'))
else:
    experimental_annots = {}
    bar1 = tqdm(total=len(validation_proteins))
    for rawline in gzip.open(mf_annotations_path, 'rt').readlines():
        uniprot_id, labels = rawline.rstrip('\n').split('\t')
        if uniprot_id in validation_proteins:
            experimental_annots[uniprot_id] = labels.split(',')
            bar1.update(1)
    bar1.close()

    json.dump(experimental_annots, open(annots_cache, 'w'), indent=4)

print('Analyzing tools')
df_paths = glob(others_dir+'/*-preds.parquet')
metrics_all = {}
for df_path in tqdm(df_paths):
    metrics_all[df_path] = calc_metrics_for_tool(df_path, experimental_annots)
json.dump(metrics_all, open(f'{others_dir}/validation_results.json', 'w'), indent=4)