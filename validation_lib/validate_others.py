import gzip
import json
import polars as pl
import pandas as pd
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
import os

from stats_custom import calc_metrics_at_freq_threshold

def calc_metrics_for_tool(df_path, experimental_annots, max_n=17000):
    print(df_path)
    print('Loading')
    labels_path = df_path.replace('-preds.parquet', '-label_names.txt')
    labels_sequence = open(labels_path, 'r').read().split('\n')
    df = pl.read_parquet(df_path)
    ids_list = df['id'].to_list()
    true_labels = []
    #print(np.array(labels_sequence))
    
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
    #print(val_y_pred)
    col_sums = true_labels.sum(axis=0)
    print(max(col_sums))
    print(len([1 for s in col_sums if s == true_labels.shape[0]]))
    print(true_labels.shape)
    #quit(1)
    #freq_thresholds = [9, 7, 5, 3, 1, 0]
    freq_thresholds = [3, 0]
    #freq_thresholds = [6]
    for min_freq in freq_thresholds:
        print('min freq:', min_freq)
        print(f"Removed {np.sum(~(col_sums > min_freq))} columns with low frequency in true_labels.")
    metrics_dict = {}
    
    for min_freq in freq_thresholds:
        new_m = calc_metrics_at_freq_threshold(col_sums, true_labels, val_y_pred, 
            min_freq, labels_sequence)
        metrics_dict[f'Min_Freq_{min_freq+1}'] = new_m
        print(json.dumps(new_m, indent=2))
    print(metrics_dict)
    return metrics_dict

def make_metrics_df(metrics_all):
    curve_metrics = ["ROC AUC", "ROC AUC W", "AUPRC", "AUPRC W", "Fmax", "Best F1 Threshold"]
    bool_metrics = ["F1 W", "Precision W", "Recall W", 
                    "Fmax", "ROC AUC", "ROC AUC W", "AUPRC", "AUPRC W"]
    metrics_dicts = []
    for p, by_th in metrics_all.items():
        for th, metric_vals in by_th.items():
            name = os.path.basename(p) + ' - Min Freq ' + str(th)
            metrics_dicts.append((name, metric_vals))
    
    lines = []
    for name, d in metrics_dicts:
        print('Metrics for', name)
        print(d)
        line = {'Name': name}
        
        #complete metrics set
        for metric in curve_metrics:
            line['Curve '+metric] = d[metric]
        
        for metric in bool_metrics:
            if 'boolean' in d:
                line['Discrete '+metric] = d['boolean'][metric]
            else:
                line['Discrete '+metric] = None
        line_round = {}
        for k, v in line.items():
            if isinstance(v, float):
                line_round[k] = round(v*100, 2)
            else:
                line_round[k] = v
        lines.append(line_round)
    df = pl.DataFrame(lines)
    print(df)
    return df

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

validation_results_path = f'{others_dir}/validation_results.json'
if not os.path.exists(validation_results_path):
    print('Analyzing tools')
    df_paths = glob(others_dir+'/*-preds.parquet')
    metrics_all = {}
    for df_path in tqdm(df_paths):
        metrics_all[df_path] = calc_metrics_for_tool(df_path, experimental_annots)
    json.dump(metrics_all, open(validation_results_path, 'w'), indent=4)
else:
    metrics_all = json.load(open(validation_results_path, 'r'))

'''df_lines = []
col = ['Fmax','AUPRC W','ROC AUC W','Tool Labels','Evaluated Labels','Best F1 Threshold','ROC AUC','AUPRC','N Proteins','bases']
for p, by_th in metrics_all.items():
    for th, metric_vals in by_th.items():
        new_line = [os.path.basename(p), str(th)] + [str(metric_vals[c]) for c in col]
        df_lines.append(new_line)

df_final = pd.DataFrame(df_lines)
df_final.columns = ['Software', 'Min. GO ID Freq.'] + col'''
df_final = make_metrics_df(metrics_all)
print('Saving results')
df_final.write_csv(f'{others_dir}/validation_results.csv', separator='\t')
