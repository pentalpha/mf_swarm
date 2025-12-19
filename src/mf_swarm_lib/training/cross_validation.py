from glob import glob
import json
from os import mkdir, path
import os
import sys
import numpy as np
from pickle import load, dump

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mf_swarm_lib.core.ml.custom_statistics import faster_fmax, norm_with_baseline
from sklearn import metrics

from mf_swarm_lib.utils.util_base import concat_lists, run_command
from mf_swarm_lib.data.parquet_loading import load_columns_from_parquet
from mf_swarm_lib.core.ml.multi_input_clf import makeMultiClassifierModel, MultiInputNet
from mf_swarm_lib.core.ensemble import BasicEnsemble
#from mf_swarm_lib.core.node_factory import makeMultiClassifierModel
import polars as pl
    
# Removed prepare_features_folds as it is now handled globally during dataset generation

def prepare_labels_folds(labels_path, n_folds, max_proteins=60000):
    base_dir = labels_path.replace('.parquet', '_folds'+str(n_folds))
    
    fold_ids = [str(i) for i in range(n_folds)]
    train_y_basename = base_dir + '/train_y.obj'
    test_y_basename = base_dir + '/test_y.obj'

    flows = {fold_i: {
            'train_y': train_y_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_y': test_y_basename.replace('.obj', '_'+fold_i+'.obj')
            }
        for fold_i in fold_ids}

    all_paths = concat_lists([list(f.values()) for f in flows.values()])
    if all([path.exists(p) for p in all_paths]):
        return base_dir
    
    print('Preparing Labels Folds', base_dir)
    run_command(['mkdir -p', base_dir])
    
    cols_to_use = ['id', 'fold_id', 'labels']
    labels_df = pl.read_parquet(labels_path, columns=cols_to_use)

    print('Separating splits', file=sys.stderr)
    traintest_parts = []
    for fold_idx in range(n_folds):
        part = labels_df.filter(pl.col('fold_id') == fold_idx)
        traintest_parts.append(part)

    print('Using splits to create train/test folds', file=sys.stderr)
    for test_fold_i, fold_parts_dict in flows.items():
        train_parts = [part 
            for i, part in enumerate(traintest_parts) 
            if i != int(test_fold_i)]
        test = traintest_parts[int(test_fold_i)]
        train = pl.concat(train_parts)
    
        train_y_np = train['labels'].to_numpy()
        test_y_np = test['labels'].to_numpy()
        
        to_save = [(fold_parts_dict['train_y'], train_y_np),
                (fold_parts_dict['test_y'], test_y_np)]
        
        if not path.exists('tmp'):
            run_command(['mkdir', 'tmp'])
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))
    
    return base_dir

# Trains a node using cross-validation, returning an ensemble of models
def train_crossval_node(params: dict, features: list, n_folds: int, max_proteins=60000):
    node = params['node']
    params_dict = params['params_dict']

    labels_path = node['traintest_labels_path']
    node_name = params['node_name']
    dirname = path.dirname(labels_path)
    features_base_dir = path.join(dirname, 'features_traintest_folds' + str(n_folds))
    labels_base_dir = path.join(dirname, 'labels_traintest_' + node_name + '_folds' + str(n_folds))

    assert path.isdir(labels_base_dir)
    assert path.isdir(features_base_dir)

    models = []
    stats_dicts = []
    for fold_i in range(n_folds):
        fold_i_str = str(fold_i)

        train_x_name = features_base_dir + '/train_x_'+fold_i_str+'.obj'
        test_x_name = features_base_dir + '/test_x_'+fold_i_str+'.obj'
        
        train_y_name = labels_base_dir + '/train_y_'+fold_i_str+'.obj'
        test_y_name = labels_base_dir + '/test_y_'+fold_i_str+'.obj'

        assert path.exists(train_x_name)
        assert path.exists(test_x_name)
        assert path.exists(train_y_name)
        assert path.exists(test_y_name)

        # Load global generic feature folds
        train_x_all = load(open(train_x_name, 'rb'))
        test_x_all = load(open(test_x_name, 'rb'))
        
        # Filter to requested features
        train_x = [(name, arr) for name, arr in train_x_all if name in features]
        test_x = [(name, arr) for name, arr in test_x_all if name in features]
        
        train_y = load(open(train_y_name, 'rb'))
        test_y = load(open(test_y_name, 'rb'))

        annot_model, stats = makeMultiClassifierModel(train_x, train_y, test_x, test_y, 
            params_dict)
        models.append(annot_model)
        stats_dicts.append(stats)
        print(stats)

    annot_model = BasicEnsemble(models, stats_dicts)
    print('Ensemble stats:',annot_model.stats)

    return annot_model

class CrossValRunner():
    def __init__(self, param_translator, node, features, n_folds):
        self.new_param_translator = param_translator
        self.node = node
        self.features = features
        self.n_folds = n_folds
        
    def objective_func(self, solution):
        #print('objective_func', file=sys.stderr)
        new_params_dict = self.new_param_translator.decode(solution)
        self.node['params_dict'] = new_params_dict
        #print('getting roc_auc s', file=sys.stderr)
        model_obj = train_crossval_node(self.node, self.features, n_folds=self.n_folds)
        print(model_obj.stats)
        #print(roc_aucs, file=sys.stderr)
        #print('objective_func finish', file=sys.stderr)
        return model_obj.stats

def validate_cv_model_noretrain(annot_model, val_labels_path, go_labels, features, baseline_values):
    # Derive features path
    # val_labels_path: .../labels_val_<cluster>.parquet
    # features_path: .../features_val.parquet
    dirname = path.dirname(val_labels_path)
    features_path = path.join(dirname, 'features_val.parquet')

    labels_df = pl.read_parquet(val_labels_path, columns=['id', 'labels'])
    features_df = pl.read_parquet(features_path, columns=['id'] + features)
    
    # Join on id
    val_df = labels_df.join(features_df, on='id', how='inner')
    
    # Ensure order logic if needed, but 'inner' join usually keeps reasonable order. 
    # Or just use the DF as is.
    
    val_x_np = []
    for col in features:
        val_x_np.append(val_df[col].to_numpy())
    val_df_y_np = val_df['labels'].to_numpy()
    roc_auc_score_mac_base = baseline_values['roc_auc_score_mac']
    roc_auc_score_w_base = baseline_values['roc_auc_score_w']
    auprc_mac_base = baseline_values['auprc_mac']
    auprc_w_base = baseline_values['auprc_w']

    validations = []
    test_stats_vec = annot_model.stats_dicts + [annot_model.stats]
    basemodels_and_ensemble = annot_model.models + [annot_model]
    val_y_pred = None
    for model, stats in zip(basemodels_and_ensemble, test_stats_vec):
        val_y_pred = model.predict(val_x_np, verbose=0)
        fmax, bestrh = faster_fmax(val_y_pred, val_df_y_np)
        roc_auc_score_mac = metrics.roc_auc_score(val_df_y_np, val_y_pred, average='macro')
        roc_auc_score_w = metrics.roc_auc_score(val_df_y_np, val_y_pred, average='weighted')
        auprc_mac = metrics.average_precision_score(val_df_y_np, val_y_pred)
        auprc_w = metrics.average_precision_score(val_df_y_np, val_y_pred, average='weighted')
        roc_auc_score_mac_norm = norm_with_baseline(roc_auc_score_mac, roc_auc_score_mac_base)
        roc_auc_score_w_norm = norm_with_baseline(roc_auc_score_w, roc_auc_score_w_base)
        auprc_mac_norm = norm_with_baseline(auprc_mac, auprc_mac_base)
        auprc_w_norm = norm_with_baseline(auprc_w, auprc_w_base)
        
        val_stats = {
            'raw':{
                'ROC AUC': float(roc_auc_score_mac),
                'ROC AUC W': float(roc_auc_score_w),
                'AUPRC': float(auprc_mac),
                'AUPRC W': float(auprc_w),
            },
            'ROC AUC': float(roc_auc_score_mac_norm),
            'ROC AUC W': float(roc_auc_score_w_norm),
            'AUPRC': float(auprc_mac_norm),
            'AUPRC W': float(auprc_w_norm),
            'Fmax': float(fmax),
            'Best Fmax Threshold': float(bestrh),
            'val_x': len(val_df),
            'quickness': stats['quickness']
        }

        metric_weights = [('Fmax', 4), ('ROC AUC W', 4), ('AUPRC W', 4), 
                        ('quickness', 2)]
        w_total = sum([w for m, w in metric_weights])
        val_stats['fitness'] = sum([val_stats[m]*w for m, w in metric_weights]) / w_total
        validations.append(val_stats)
    
    results = {
        'test': annot_model.stats,
        'validation': validations[-1],
        'base_model_validations': validations[:len(validations)-1],
        'go_labels': go_labels,
    }

    validation_solved_df = pl.DataFrame(
        {
            'id': val_df['id'].to_list(),
            'y': val_df_y_np,
            'y_pred': val_y_pred
        }
    )

    return results, validation_solved_df 

def validate_cv_model(node, solution_dict, features, n_folds=5):
    node['params_dict'] = solution_dict
    annot_model = train_crossval_node(node, features, n_folds=n_folds)
    print('Validating')
    val_path = node['node']['val_labels_path']
    baseline_values = node['node']['baseline_metrics']
    go_labels = node['node']['go']
    
    results, validation_solved_df = validate_cv_model_noretrain(
        annot_model, val_path, go_labels, features,
        baseline_values=baseline_values)
    
    return annot_model, results, validation_solved_df