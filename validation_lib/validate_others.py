import gzip
import json
import multiprocessing
import polars as pl
import pandas as pd
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
import os

from stats_custom import calc_metrics_at_freq_threshold
from go_expansion import gos_not_to_use, load_go_graph, create_expanded_df

freq_thresholds = [3]
N_PROCS = 7

def calc_metrics_for_tool(params_dict):
    #{'df_path': df_path, 'experimental_annots': experimental_annots}
    df_path = params_dict['df_path']
    experimental_annots = params_dict['experimental_annots']
    max_n=17000
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
    raw_metrics = ["ROC AUC", "ROC AUC W", "AUPRC", "AUPRC W"]
    bool_metrics = ["F1 W", "Precision W", "Recall W",
                    "F1", "Precision", "Recall", 
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
        line = {'Name': name, 'N Proteins': d['N Proteins'], 
            'Evaluated Labels': d['Evaluated Labels'],
            'Tool Labels': d['Tool Labels']}
        
        for metric in raw_metrics:
            line['Raw '+metric] = d[metric]
        
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
    df = df.sort(by=['Curve Fmax', 'Discrete Precision W'])
    print(df)
    return df

def create_expanded_dfs(df_paths, expanded_dir, go_graph_path):
    if not os.path.exists(expanded_dir):
        os.mkdir(expanded_dir)
    
    modes = ['mean', 'min', 'max']
    not_use = gos_not_to_use()
    G = load_go_graph(go_graph_path)
    expanded_paths = []
    new_df_params = []
    for df_path in df_paths:
        go_ids_path = df_path.replace('-preds.parquet', '-label_names.txt')
        for mode in modes:
            output_path = expanded_dir + '/' + os.path.basename(df_path).replace('-preds', '_'+mode+'-preds')
            params = {'df_path': df_path, 'mode': mode, 'G': G, 
                'goids_path': go_ids_path, 'not_use': not_use, 
                'output_path': output_path}
            expanded_paths.append(output_path)
            if not os.path.exists(output_path):
                new_df_params.append(params)
    with multiprocessing.Pool(N_PROCS) as pool:
        pool.map(create_expanded_df, new_df_params)
    return expanded_paths

if __name__ == "__main__":
    #sys.argv = ['others/validate_others.py', '~/data/protein_dimension_db/release_1/', 
    #    '40', '0.15', '~/data/mf_swarm_datasets/other_tools/']
    prot_dimension_db_release_path = os.path.expanduser(sys.argv[1])
    min_protein_annots = sys.argv[2]
    val_part = sys.argv[3]
    others_dir = os.path.expanduser(sys.argv[4])

    df_paths = glob(others_dir+'/*-preds.parquet')
    '''go_graph_path = prot_dimension_db_release_path + '/go-basic.obo'
    expanded_dir = others_dir + '/expanded_dfs'
    expanded_dfs = create_expanded_dfs(df_paths, expanded_dir, go_graph_path)
    df_paths = df_paths + expanded_dfs'''

    validation_results_path = f'{others_dir}/validation_results.json'
    metrics_all = json.load(open(validation_results_path, 'r')) if os.path.exists(validation_results_path) else {}
    dfs_to_calc = [p for p in df_paths if not p in metrics_all]
    if len(dfs_to_calc) > 0:
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
        metrics_calc_params = []
        for df_path in tqdm(dfs_to_calc):
            metrics_calc_params.append({'df_path': df_path, 'experimental_annots': experimental_annots})
        print('Analyzing tools')
        with multiprocessing.Pool(N_PROCS) as pool:
            new_metric_entries = pool.map(calc_metrics_for_tool, metrics_calc_params)
            for df_path, new_vals in zip(dfs_to_calc, new_metric_entries):
                metrics_all[df_path] = new_vals
        
    json.dump(metrics_all, open(validation_results_path, 'w'), indent=4)

    df_final = make_metrics_df(metrics_all)
    if len(freq_thresholds) == 1:
        default_freq = freq_thresholds[0]+1
        df_final = df_final.filter(pl.col('Name').str.contains(f'{default_freq}'))
        names_original = df_final['Name'].to_list()
        new_names = [name.split(' - Min')[0] for name in names_original]
        new_names = [name.replace('-preds.parquet', '').replace('_validation', '') for name in new_names]
        new_names = [name.strip().upper() for name in new_names]
        df_final = df_final.with_columns(pl.Series('Name', new_names))
    print(df_final)
    print('Saving results')
    df_final.write_csv(f'{others_dir}/validation_results.csv', separator='\t')

    scatter_axis = ['Discrete Precision W', 'Discrete Recall W']
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(5,5))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    xvals = df_final[scatter_axis[0]].to_list()
    yvals = df_final[scatter_axis[1]].to_list()
    names = df_final['Name'].to_list()
    #
    ax.scatter(xvals, yvals, s=100, c='orange', edgecolors='black', alpha=0.9)
    for i, name in enumerate(names):
        #central alignment, above the point
        va = 'bottom'
        #except if its TALE, then put below
        offset = 6
        if name == 'TALE':
            va = 'top'
            offset = -6
        #with a little offset from the y point
        ax.annotate(name, (xvals[i], yvals[i]), ha='center', 
                    va=va, xytext=(0, offset), textcoords='offset points')
    ax.set_xlabel('Precision (%)')
    ax.set_ylabel('Recall (%)')
    #both axis are represented between 40% and 100%
    ax.set_xlim(40, 100)
    ax.set_ylim(40, 100)
    ax.set_title('Molecular Function Prediction:\nPrecision VS Recall')
    #remove the right spine
    ax.spines['right'].set_visible(False)
    #remove the top spine
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{others_dir}/precision_recall_scatter.png', dpi=200)