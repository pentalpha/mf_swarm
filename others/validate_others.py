import gzip
import json
import polars as pl
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
from sklearn import metrics
from sklearn import preprocessing

def norm_with_baseline(metric_val, metric_baseline, max_value = 1.0):
    actual_range = max_value - metric_baseline
    actual_zero = metric_baseline
    metric_norm = (metric_val - actual_zero) / actual_range
    return metric_norm

if __name__ == '__main__':
    prot_dimension_db_release_path = sys.argv[1]
    min_protein_annots = sys.argv[2]
    val_part = sys.argv[3]
    others_dir = sys.argv[4]
    
    print('Loading validation proteins')
    validation_split_dir = prot_dimension_db_release_path + f'/validation_splits/min_prot{min_protein_annots}_val{val_part}'
    val_ids = validation_split_dir+'/validation_ids.txt'
    validation_proteins = open(val_ids, 'r').read().split('\n')

    print('Loading experimental annotations')
    mf_annotations_path = f'{prot_dimension_db_release_path}/go.experimental.mf.tsv.gz'
    annots_cache = f'{others_dir}/experimental_annots.json'
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

        true_labels = np.asarray(true_labels)
        col_sums = true_labels.sum(axis=0)
        freq_thresholds = [9, 7, 5, 3, 1, 0]
        for min_freq in freq_thresholds:
            print('min freq:', min_freq)
            print(f"Removed {np.sum(~(col_sums > min_freq))} columns with zero frequency in true_labels.")
        metrics_dict = {}
        val_y_pred = df['labels'].to_numpy()
        
        for min_freq in freq_thresholds:
            print('min freq:', min_freq)
            valid_cols = col_sums > min_freq

            true_labels_f = true_labels[:, valid_cols]
            val_y_pred_f = val_y_pred[:, valid_cols]
            baseline_pred_f = np.random.rand(*val_y_pred_f.shape)

            #min_max_scaler = preprocessing.MinMaxScaler()
            #val_y_pred_f_scaled = min_max_scaler.fit_transform(val_y_pred_f)
            labels_sequence_f = [label for i, label in enumerate(labels_sequence) if valid_cols[i]]

            print(f"Removed {np.sum(~valid_cols)} columns with zero frequency in true_labels.")

            evaluated_label_sequence_path = labels_path.replace('.txt', f'.min_{min_freq}.evaluated.txt')
            open(evaluated_label_sequence_path, 'w').write('\n'.join(labels_sequence_f))

            print('Comparing')
            roc_auc_score_mac = metrics.roc_auc_score(true_labels_f, val_y_pred_f, average='macro')
            roc_auc_score_mac_base = metrics.roc_auc_score(true_labels_f, baseline_pred_f, average='macro')
            roc_auc_score_mac_norm = norm_with_baseline(roc_auc_score_mac, roc_auc_score_mac_base)
            print(roc_auc_score_mac, roc_auc_score_mac_norm)
            roc_auc_score_w = metrics.roc_auc_score(true_labels_f, val_y_pred_f, average='weighted')
            roc_auc_score_w_base = metrics.roc_auc_score(true_labels_f, baseline_pred_f, average='weighted')
            roc_auc_score_w_norm = norm_with_baseline(roc_auc_score_w, roc_auc_score_w_base)
            print(roc_auc_score_w, roc_auc_score_w_norm)
            auprc_mac = metrics.average_precision_score(true_labels_f, val_y_pred_f)
            auprc_mac_base = metrics.average_precision_score(true_labels_f, baseline_pred_f)
            auprc_mac_norm = norm_with_baseline(auprc_mac, auprc_mac_base)
            print(auprc_mac, auprc_mac_norm)
            auprc_w = metrics.average_precision_score(true_labels_f, val_y_pred_f, average='weighted')
            auprc_w_base = metrics.average_precision_score(true_labels_f, baseline_pred_f, average='weighted')
            auprc_w_norm = norm_with_baseline(auprc_w, auprc_w_base)
            print(auprc_w, auprc_w_norm)
            acc = np.mean(keras.metrics.binary_accuracy(true_labels_f, val_y_pred_f).numpy())
            acc_base = np.mean(keras.metrics.binary_accuracy(true_labels_f, baseline_pred_f).numpy())
            acc_norm = norm_with_baseline(acc, acc_base)
            new_m = {
                'raw':{
                    'ROC AUC': float(roc_auc_score_mac),
                    'ROC AUC W': float(roc_auc_score_w),
                    'AUPRC': float(auprc_mac),
                    'AUPRC W': float(auprc_w),
                    'Accuracy': float(acc), 
                },
                'norm': {
                    'ROC AUC': float(roc_auc_score_mac_norm),
                    'ROC AUC W': float(roc_auc_score_w_norm),
                    'AUPRC': float(auprc_mac_norm),
                    'AUPRC W': float(auprc_w_norm),
                    'Accuracy': float(acc_norm),
                    'bases': {
                        'ROC AUC': float(roc_auc_score_mac_base),
                        'ROC AUC W': float(roc_auc_score_w_base),
                        'AUPRC': float(auprc_mac_base),
                        'AUPRC W': float(auprc_w_base),
                        'Accuracy': float(acc_base),
                    }
                },
                'N Proteins': len(true_labels_f),
                'Tool Labels': len(valid_cols),
                'Evaluated Labels': len(labels_sequence_f)
            }
            metrics_dict[f'Min_Freq_{min_freq+1}'] = new_m
            print(json.dumps(new_m, indent=2))
        print(metrics_dict)
        metrics_all[df_path] = metrics_dict
    
    json.dump(metrics_all, open(f'{others_dir}/validation_results.json', 'w'), indent=4)