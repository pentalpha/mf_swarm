import numpy as np
from glob import glob
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score

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
def faster_fmax(pred_scores, truth_set, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 80)
    has_positives = np.any(truth_set, axis=1)  # Boolean mask (n_samples,)
    pred_scores_filtered = pred_scores[has_positives]
    truth_set_filtered = truth_set[has_positives]
    # pred_scores_filtered: (n_samples, n_labels)
    # truth_set_filtered: (n_samples, n_labels)
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

def calc_metrics_at_freq_threshold(col_sums, true_labels, val_y_pred, 
        min_freq, labels_sequence, labels_path):
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
    true_labels_f = true_labels[:, valid_cols]
    val_y_pred_f = val_y_pred[:, valid_cols]
    baseline_pred_f = np.random.rand(*val_y_pred_f.shape)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #val_y_pred_f_scaled = min_max_scaler.fit_transform(val_y_pred_f)
    labels_sequence_f = [label for i, label in enumerate(labels_sequence) if valid_cols[i]]

    print(f"Removed {np.sum(~valid_cols)} columns with low frequency in true_labels.")

    evaluated_label_sequence_path = labels_path.replace('.txt', f'.min_{min_freq}.evaluated.txt')
    open(evaluated_label_sequence_path, 'w').write('\n'.join(labels_sequence_f))

    print('Comparing')
    fmax, bestrh = faster_fmax(val_y_pred_f, true_labels_f)
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
        'N Proteins': len(true_labels_f),
        'Tool Labels': len(valid_cols),
        'Evaluated Labels': len(labels_sequence_f)
    }
    
    return new_m