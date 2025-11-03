import json
from os import path
from glob import glob

from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import polars as pl

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

def load_swarm_params_and_results_jsons(full_swarm_exp_dir):
    node_dirs = glob(full_swarm_exp_dir + '/Level-*')
    node_dicts = [x+'/exp_params.json' for x in node_dirs]
    node_results = [x+'/exp_results.json' for x in node_dirs]

    params_jsons = []
    results_jsons = []
    for exp_params, exp_results in zip(node_dicts, node_results):
        if not path.exists(exp_params):
            params_jsons.append(exp_params.replace('exp_params.json', 'standard_params.json'))
        else:
            params_jsons.append(exp_params)
        std_results = exp_results.replace('exp_results.json', 'standard_results.json')
        if path.exists(exp_results):
            results_jsons.append(exp_results)
        elif path.exists(std_results):
            results_jsons.append(std_results)
        else:
            results_jsons.append(None)

    return params_jsons, results_jsons

def draw_cv_relevance(full_swarm_exp_dir: str, output_dir: str):
    params_jsons, results_jsons = load_swarm_params_and_results_jsons(full_swarm_exp_dir)
    
    #n_proteins = []
    #node_names = []
    auprc_difs = []
    roc_auc_difs = []
    for exp_params, exp_results in zip(params_jsons, results_jsons):
        params = json.load(open(exp_params, 'r'))
        if exp_results is not None:
            results = json.load(open(exp_results, 'r'))
            auprc_w = results['validation']['AUPRC W']*100
            roc_auc_w = results['validation']['ROC AUC W']*100

            base_auprc_ws = [x['AUPRC W']*100 for x in results['base_model_validations']]
            base_roc_auc_ws = [x['ROC AUC W']*100 for x in results['base_model_validations']]

            difs = [auprc_w - x for x in base_auprc_ws]
            difs2 = [roc_auc_w - x for x in base_roc_auc_ws]

            auprc_difs.extend(difs)
            roc_auc_difs.extend(difs2)
            #auprc_difs.append(auprc_w - max(base_auprc_ws))
            #roc_auc_difs.append(roc_auc_w - max(base_roc_auc_ws))
    print(auprc_difs)
    print(roc_auc_difs)
    #Create box plot of AUPRC diferences and ROC AUC differences using matplotlib
    plt.figure(figsize=(5, 8))
    plt.boxplot([auprc_difs, roc_auc_difs], labels=['AUPRC Gains', 'ROC AUC Gains'])
    plt.title('Classification Performance Gains from Cross-Validation')
    plt.ylabel('Difference (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.join(output_dir, 'cv_relevance_boxplot.png'))
    plt.close()

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
        if any([x not in [0.0, 1.0] for x in col_scores]):
            # If there are non-binary scores, we can find a threshold
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
        else:
            # If all scores are binary, set threshold to 0.5
            best_thresholds[col_ids[i]] = 0.5
            print(f"All scores for {col_ids[i]} are binary. Setting threshold to 0.5")
        
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
    true_labels_f = df[truth_col].to_numpy()
    val_y_pred_f = df[scores_col].to_numpy()

    return eval_predictions( 
        true_labels_f, val_y_pred_f, 
        go_id_sequence=go_id_sequence,
        thresholds=thresholds)
    

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