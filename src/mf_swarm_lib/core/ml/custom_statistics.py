import json

import numpy as np
from tqdm import tqdm
import polars as pl
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics

#SKLearn implementation of the Fmax metric
def fast_fmax(pred_scores, truth_set, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 80)
        print('thresholds:', thresholds)
    best_fmax = 0.0
    best_th = 0.0
    for th in tqdm(thresholds):
        pred_bin = (pred_scores > th).astype(int)
        # Vetorizado: calcula média por proteína (sample average)
        rec = recall_score(truth_set, pred_bin, average='samples', zero_division=0)
        prec = precision_score(truth_set, pred_bin, average='samples', zero_division=0)
        if rec + prec > 0:
            f = 2 * prec * rec / (prec + rec)
        else:
            f = 0.0
        if f > best_fmax:
            best_fmax = f
            best_th = th
            print(best_fmax, best_th)
    return best_fmax, best_th

#Numpy implementation of the Fmax metric
def faster_fmax(pred_scores, truth_set, n_ths=120):
    thresholds = np.linspace(0, 1, n_ths)
    # pred_scores: (n_samples, n_labels)
    # truth_set: (n_samples, n_labels)
    has_positives = np.any(truth_set, axis=1)  # Boolean mask (n_samples,)
    pred_scores_filtered = pred_scores[has_positives]
    truth_set_filtered = truth_set[has_positives]
    n_samples, n_labels = pred_scores_filtered.shape
    n_thresh = len(thresholds)
    # Cria matriz booleana para todos os thresholds: shape (n_thresh, n_samples, n_labels)
    pred_bin = (pred_scores_filtered[None, :, :] > thresholds[:, None, None]).astype(np.uint8)
    truth = truth_set_filtered[None, :, :].astype(np.uint8)

    # True positives, false positives, false negatives para cada threshold
    tp = np.sum(pred_bin & truth, axis=2)  # shape (n_thresh, n_samples)
    fp = np.sum(pred_bin & (~truth), axis=2)
    fn = np.sum((~pred_bin) & truth, axis=2)

    # Precisão e recall por amostra e threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0)

    # Média por amostra (samples), depois média por threshold
    prec_mean = np.mean(prec, axis=1)
    rec_mean = np.mean(rec, axis=1)

    # F-score para cada threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(prec_mean + rec_mean > 0, 2 * prec_mean * rec_mean / (prec_mean + rec_mean), 0)

    best_idx = np.argmax(f)
    best_fmax = f[best_idx]
    best_th = thresholds[best_idx]
    return float(best_fmax), float(best_th)

def norm_with_baseline(metric_val, metric_baseline, max_value = 1.0):
    actual_range = max_value - metric_baseline
    actual_zero = metric_baseline
    metric_norm = (metric_val - actual_zero) / actual_range
    return metric_norm

def create_random_baseline(val_y_pred) -> np.ndarray:
    return np.random.rand(*val_y_pred.shape)

def find_best_threshold_per_col(scores_matrix: np.ndarray, 
        labels_matrix: np.ndarray, col_ids: list) -> dict:
    # scores_matrix: (n_samples, n_labels)
    # labels_matrix: (n_samples, n_labels)
    # col_ids: list of label names
    if scores_matrix.shape != labels_matrix.shape:
        raise ValueError("Scores and labels matrices must have the same shape.")
    n_samples, n_labels = scores_matrix.shape
    if len(col_ids) != n_labels:
        raise ValueError("Column IDs must match the number of labels in the matrices.")
    best_thresholds = {}
    for i in tqdm(range(n_labels)):
        col_scores = scores_matrix[:, i]
        col_labels = labels_matrix[:, i]
        thresholds = np.linspace(0, 1, 150)
        best_f1 = 0.0
        best_th = 0.0
        for th in thresholds:
            pred_bin = (col_scores > th).astype(int)
            rec = recall_score(col_labels, pred_bin, zero_division=0)
            prec = precision_score(col_labels, pred_bin, zero_division=0)
            if rec + prec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[col_ids[i]] = best_th
        print(f"Best threshold for {col_ids[i]}: {best_th} (F1: {best_f1})")
    return best_thresholds

def eval_predictions_dataset_bool(preds_matrix_bool: np.ndarray, 
    labels_matrix_bool: np.ndarray) -> dict:
    # preds_matrix_bool: (n_samples, n_labels)
    # labels_matrix_bool: (n_samples, n_labels)
    n_samples, n_labels = preds_matrix_bool.shape
    if preds_matrix_bool.shape != labels_matrix_bool.shape:
        raise ValueError("Predictions and labels matrices must have the same shape.")
    # True positives, false positives, false negatives para cada threshold
    f1_macro = metrics.f1_score(labels_matrix_bool, preds_matrix_bool, average='macro')
    f1_weighted = metrics.f1_score(labels_matrix_bool, preds_matrix_bool, average='weighted')
    f1_samples = metrics.f1_score(labels_matrix_bool, preds_matrix_bool, average='samples')
    precision_macro = metrics.precision_score(labels_matrix_bool, preds_matrix_bool, average='macro')
    precision_weighted = metrics.precision_score(labels_matrix_bool, preds_matrix_bool, average='weighted')
    precision_samples = metrics.precision_score(labels_matrix_bool, preds_matrix_bool, average='samples')
    recall_samples = metrics.recall_score(labels_matrix_bool, preds_matrix_bool, average='samples')
    recall_weighted = metrics.recall_score(labels_matrix_bool, preds_matrix_bool, average='weighted')
    recall_macro = metrics.recall_score(labels_matrix_bool, preds_matrix_bool, average='macro')
    f_max = faster_fmax(preds_matrix_bool, labels_matrix_bool)

    return {
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1': f1_macro,
        'Precision W': precision_weighted,
        'Recall W': recall_weighted,
        'F1 W': f1_weighted,
        'Precision S': precision_samples,
        'Recall S': recall_samples,
        'F1 S': f1_samples,
        'Fmax': f_max[0],
        'Best F1 Threshold': f_max[1],
        'N Proteins': n_samples,
        'ROC AUC': metrics.roc_auc_score(labels_matrix_bool, preds_matrix_bool, average='macro'),
        'ROC AUC W': metrics.roc_auc_score(labels_matrix_bool, preds_matrix_bool, average='weighted'),
        'AUPRC': metrics.average_precision_score(labels_matrix_bool, preds_matrix_bool, average='macro'),
        'AUPRC W': metrics.average_precision_score(labels_matrix_bool, preds_matrix_bool, average='weighted'),
    }

def convert_using_col_thresholds(scores_matrix: np.ndarray, 
                                 col_thresholds: dict, col_ids: list) -> np.ndarray:
    bool_pred_lines = []
    for protein_index in range(scores_matrix.shape[0]):
        bool_pred_line = np.array([scores_matrix[protein_index, i] > col_thresholds[col_ids[i]] 
                            for i in range(len(col_ids))])
        bool_pred_lines.append(bool_pred_line)
    val_y_pred_bool = np.asarray(bool_pred_lines)
    return val_y_pred_bool

def eval_predictions(true_labels_f, val_y_pred_f, 
        go_id_sequence=None,
        thresholds=None):
    
    baseline_pred_f = np.random.rand(*val_y_pred_f.shape)

    print('True labels:')
    print(true_labels_f)
    print('Scores:')
    print(val_y_pred_f)
    print('Baseline:')
    print(baseline_pred_f)
    print('Comparing')
    fmax, bestrh = faster_fmax(val_y_pred_f, true_labels_f)
    print('fmax', fmax, 'bestrh', bestrh)

    if go_id_sequence != None and thresholds == None:
        thresholds = find_best_threshold_per_col(val_y_pred_f, true_labels_f, 
            go_id_sequence)
    
    if go_id_sequence != None and thresholds != None:
        # If go_id_sequence and thresholds are provided, use them to filter scores
        val_y_pred_bool = convert_using_col_thresholds(val_y_pred_f, thresholds, go_id_sequence)
        bool_metrics = eval_predictions_dataset_bool(val_y_pred_bool, true_labels_f > 0)
    else:
        val_y_pred_bool = val_y_pred_f > bestrh
        bool_metrics = eval_predictions_dataset_bool(val_y_pred_bool, true_labels_f > 0)
    print('Boolean metrics:')
    print(json.dumps(bool_metrics, indent=2))
    
    roc_auc_score_mac = metrics.roc_auc_score(true_labels_f, val_y_pred_f, average='macro')
    roc_auc_score_mac_base = metrics.roc_auc_score(true_labels_f, baseline_pred_f, average='macro')
    roc_auc_score_mac_norm = norm_with_baseline(roc_auc_score_mac, roc_auc_score_mac_base)
    print('roc_auc_score_mac', roc_auc_score_mac, roc_auc_score_mac_norm)
    roc_auc_score_w = metrics.roc_auc_score(true_labels_f, val_y_pred_f, average='weighted')
    roc_auc_score_w_base = metrics.roc_auc_score(true_labels_f, baseline_pred_f, average='weighted')
    roc_auc_score_w_norm = norm_with_baseline(roc_auc_score_w, roc_auc_score_w_base)
    print('roc_auc_score_w', roc_auc_score_w, roc_auc_score_w_norm)
    auprc_mac = metrics.average_precision_score(true_labels_f, val_y_pred_f)
    auprc_mac_base = metrics.average_precision_score(true_labels_f, baseline_pred_f)
    auprc_mac_norm = norm_with_baseline(auprc_mac, auprc_mac_base)
    print('auprc_mac', auprc_mac, auprc_mac_norm)
    auprc_w = metrics.average_precision_score(true_labels_f, val_y_pred_f, average='weighted')
    auprc_w_base = metrics.average_precision_score(true_labels_f, baseline_pred_f, average='weighted')
    auprc_w_norm = norm_with_baseline(auprc_w, auprc_w_base)
    print('auprc_w', auprc_w, auprc_w_norm)
    new_m = {
        'raw':{
            'ROC AUC': float(roc_auc_score_mac),
            'ROC AUC W': float(roc_auc_score_w),
            'AUPRC': float(auprc_mac),
            'AUPRC W': float(auprc_w)
        },
        'boolean': bool_metrics,
        'ROC AUC': float(roc_auc_score_mac_norm),
        'ROC AUC W': float(roc_auc_score_w_norm),
        'AUPRC': float(auprc_mac_norm),
        'AUPRC W': float(auprc_w_norm),
        'Fmax': fmax,
        'Best F1 Threshold': bestrh,
        'bases': {
            'ROC AUC': float(roc_auc_score_mac_base),
            'ROC AUC W': float(roc_auc_score_w_base),
            'AUPRC': float(auprc_mac_base),
            'AUPRC W': float(auprc_w_base),
        },
        'N Proteins': true_labels_f.shape[0]
    }

    return new_m

def eval_predictions_dataset(df: pl.DataFrame, truth_col = 'labels', scores_col='scores', 
        go_id_sequence=None,
        thresholds=None) -> dict:
    ids_list = df['id'].to_list()
    true_labels_f = df[truth_col].to_numpy()
    val_y_pred_f = df[scores_col].to_numpy()

    baseline_pred_f = np.random.rand(*val_y_pred_f.shape)

    print('True labels:')
    print(true_labels_f)
    print('Scores:')
    print(val_y_pred_f)
    print('Baseline:')
    print(baseline_pred_f)
    print('Comparing')
    fmax, bestrh = faster_fmax(val_y_pred_f, true_labels_f)
    print(fmax, bestrh)

    if go_id_sequence != None and thresholds == None:
        thresholds = find_best_threshold_per_col(val_y_pred_f, true_labels_f, 
            go_id_sequence)
    
    if go_id_sequence != None and thresholds != None:
        # If go_id_sequence and thresholds are provided, use them to filter scores
        val_y_pred_bool = convert_using_col_thresholds(val_y_pred_f, thresholds, go_id_sequence)
        bool_metrics = eval_predictions_dataset_bool(val_y_pred_bool, true_labels_f > 0)
    else:
        val_y_pred_bool = val_y_pred_f > bestrh
        bool_metrics = eval_predictions_dataset_bool(val_y_pred_bool, true_labels_f > 0)
    print('Boolean metrics:')
    print(json.dumps(bool_metrics, indent=2))
    
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
    new_m = {
        'raw':{
            'ROC AUC': float(roc_auc_score_mac),
            'ROC AUC W': float(roc_auc_score_w),
            'AUPRC': float(auprc_mac),
            'AUPRC W': float(auprc_w)
        },
        'boolean': bool_metrics,
        'ROC AUC': float(roc_auc_score_mac_norm),
        'ROC AUC W': float(roc_auc_score_w_norm),
        'AUPRC': float(auprc_mac_norm),
        'AUPRC W': float(auprc_w_norm),
        'Fmax': fmax,
        'Best F1 Threshold': bestrh,
        'bases': {
            'ROC AUC': float(roc_auc_score_mac_base),
            'ROC AUC W': float(roc_auc_score_w_base),
            'AUPRC': float(auprc_mac_base),
            'AUPRC W': float(auprc_w_base),
        },
        'N Proteins': len(ids_list)
    }

    return new_m

'''
Calculates normalized hierarchical scores usint the DeePred methdology (Rifaioglu 2018).
For each score, it lists the paths from it to the root. For each path to the root, the proportion
of scores over a certain threshold is calculated. If the proportion is > 0.5, the score is
considered a positive prediction. If most predecessors are below the threshold, the score is
considered a negative prediction.

The threshold is given by the Fmax for the model.
'''
def calc_deepred_scores(raw_df: pl.DataFrame, go_id_sequence: list, 
        paths_to_root: dict, threshold: float | dict, min_prop = 0.5) -> dict:
    scores_matrix = raw_df['scores'].to_numpy()
    if isinstance(threshold, dict):
        # If threshold is a dict, convert scores using the thresholds for each GO ID
        bool_preds_matrix = convert_using_col_thresholds(scores_matrix, threshold, go_id_sequence)
    elif isinstance(threshold, float):
        # If threshold is a float, apply it to all scores
        bool_preds_matrix = scores_matrix > threshold
    else:
        raise ValueError("Threshold must be a float or a dict with GO IDs as keys.")
    #labels_matrix = raw_df['labels'].to_numpy()
    if isinstance(threshold, float):
        print('Calculating DeepPred scores for threshold:', threshold)
    else:
        print('Calculating DeepPred scores for GO ID specific thresholds:')
    print('min_prop:', min_prop )
    norm_lines = []
    for i in tqdm(range(bool_preds_matrix.shape[0])):
        has_go_id = bool_preds_matrix[i]
        positive_gos = set([go_id_sequence[j] for j, v in enumerate(has_go_id) if v])
        #correct_preds = labels_matrix[i]
        norm_line = []
        for go_id_index, go_id in enumerate(go_id_sequence):
            local_result = go_id in positive_gos
            if local_result:
                paths = paths_to_root[go_id]
                if paths:
                    path_scores = [len([g for g in p if g in positive_gos]) / len(p) 
                                for p in paths if len(p) > 0]
                    if len(path_scores) > 0:
                        if max(path_scores) <= min_prop:
                            local_result = False
            norm_line.append(local_result)
            
        norm_lines.append(np.array(norm_line))
    norm_lines = np.asarray(norm_lines)
    true_labels = raw_df['labels'].to_numpy() > 0
    deepred_metrics = eval_predictions_dataset_bool(norm_lines, true_labels)
    print(json.dumps(deepred_metrics, indent=2))
    return deepred_metrics


def calc_metrics_at_freq_threshold(col_sums, true_labels, val_y_pred, 
        min_freq, labels_sequence):
    print('min freq:', min_freq)
    valid_cols = col_sums > min_freq
    valid2 = col_sums < true_labels.shape[0]
    valid_cols = np.array([a and b for a,b in zip(valid_cols, valid2)])
    print('true_labels')
    #print(true_labels)
    print(true_labels.shape)
    print('val_y_pred')
    #print(val_y_pred)
    print(val_y_pred.shape)
    print('valid_cols')
    #print(valid_cols)
    print(valid_cols.shape)
    print('col_sums')
    #print(col_sums)
    print(col_sums.shape)
    labels_sequence2 = [labels_sequence[i] for i in range(len(labels_sequence)) if valid_cols[i]]
    true_labels_f = true_labels[:, valid_cols]
    val_y_pred_f = val_y_pred[:, valid_cols]

    m = eval_predictions( 
        true_labels_f, val_y_pred_f, 
        go_id_sequence=labels_sequence2,
        thresholds=None)
    m['Tool Labels'] = len(labels_sequence)
    m['Evaluated Labels'] = len(labels_sequence2)
    return m