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
    
def split_train_test_n_folds(traintest_path, features, max_proteins=60000):
    #Make sure the dataset is prepared as train_x, train_y, test_x and test_y binary files
    #If not prepared, converts the parquet file to separate binary files and saves them
    fold_ids = load_columns_from_parquet(traintest_path, ['fold_id'])
    n_folds = fold_ids['fold_id'].max() + 1
    base_dir = traintest_path.replace('.parquet', '_folds'+str(n_folds)+"_features-"+'-'.join(features))
    
    fold_ids = [str(i) for i in range(n_folds)]

    train_x_basename = base_dir + '/train_x.obj'
    train_y_basename = base_dir + '/train_y.obj'
    test_x_basename = base_dir + '/test_x.obj'
    test_y_basename = base_dir + '/test_y.obj'

    folds = {fold_i: {
            'train_x': train_x_basename.replace('.obj', '_'+fold_i+'.obj'),
            'train_y': train_y_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_x': test_x_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_y': test_y_basename.replace('.obj', '_'+fold_i+'.obj')
            }
        for fold_i in fold_ids}

    all_paths = concat_lists([list(f.values()) for f in folds.values()])

    if all([path.exists(p) for p in all_paths]):
        return True
    else:
        print('Preparing', base_dir)
        run_command(['mkdir -p', base_dir])
        cols_to_use = ['id', 'fold_id'] + features + ['labels']
        traintest = load_columns_from_parquet(traintest_path, cols_to_use)
        '''if len(traintest) > max_proteins:
            print(traintest_path, 'is too large, sampling down')
            traintest = traintest.sample(fraction=(max_proteins/len(traintest)), 
                shuffle=True, seed=1337+int(fold_i))'''

        print('Separating splits', file=sys.stderr)
        # Cria os folds usando a coluna fold_id já existente
        traintest_parts = []
        for fold_idx in range(n_folds):
            part = traintest.filter(pl.col('fold_id') == fold_idx)
            traintest_parts.append(part)

        print('Using splits to create train/test folds', file=sys.stderr)
        for test_fold_i, fold_parts_dict in folds.items():
            train_parts = [part 
                for i, part in enumerate(traintest_parts) 
                if i != int(test_fold_i)]
            test = traintest_parts[int(test_fold_i)]
            train = pl.concat(train_parts)
        
            feature_columns = [c for c in train.columns if not c in ['labels', 'fold_id', 'id']]
            train_ids = train['id'].to_list()
            train_x = train.select(feature_columns)
            train_y = train.select('labels')
            test_ids = test['id'].to_list()
            test_x = test.select(feature_columns)
            test_y = test.select('labels')

            train_x_np = []
            for col in train_x.columns:
                train_x_np.append((col, train_x[col].to_numpy()))
            train_y_np = train_y['labels'].to_numpy()

            test_x_np = []
            for col in test_x.columns:
                test_x_np.append((col, test_x[col].to_numpy()))
            test_y_np = test_y['labels'].to_numpy()
            
            to_save = [(fold_parts_dict['train_x'], train_x_np), 
                    (fold_parts_dict['train_y'], train_y_np),
                    (fold_parts_dict['test_x'], test_x_np),
                    (fold_parts_dict['test_y'], test_y_np)]
            if not path.exists('tmp'):
                run_command(['mkdir', 'tmp'])
            for p, obj in to_save:
                dump(obj, open(p, 'wb'))

            open(base_dir+'/train_ids_'+test_fold_i+'.txt', 'w').write('\n'.join(train_ids))
            open(base_dir+'/test_ids_'+test_fold_i+'.txt', 'w').write('\n'.join(test_ids))
        
        return True


# Trains a node using cross-validation, returning an ensemble of models
# The node is a dictionary with the parameters and paths to the data
# The features are a list of feature names to be used in the model
# n_folds is the number of folds to be used in cross-validation
# max_proteins is the maximum number of proteins to be used in the training
# The function returns an ensemble of models trained on the cross-validation folds
# The models are trained using the makeMultiClassifierModel function
# The stats_dicts are the statistics of the models trained on each fold
def train_crossval_node(params: dict, features: list, n_folds: int, max_proteins=60000):
    node = params['node']
    params_dict = params['params_dict']

    traintest_path = node['traintest_path']
    base_dir = traintest_path.replace('.parquet', 
        '_folds'+str(n_folds)+"_features-"+'-'.join(features))
    #print('loading', base_dir)
    
    models = []
    stats_dicts = []
    for fold_i in range(n_folds):
        fold_i_str = str(fold_i)

        train_x_name = base_dir + '/train_x_'+fold_i_str+'.obj'
        train_y_name = base_dir + '/train_y_'+fold_i_str+'.obj'
        test_x_name = base_dir + '/test_x_'+fold_i_str+'.obj'
        test_y_name = base_dir + '/test_y_'+fold_i_str+'.obj'

        train_x = load(open(train_x_name, 'rb'))
        train_y = load(open(train_y_name, 'rb'))
        test_x = load(open(test_x_name, 'rb'))
        test_y = load(open(test_y_name, 'rb'))

        annot_model, stats = makeMultiClassifierModel(train_x, train_y, test_x, test_y, 
            params_dict)
        models.append(annot_model)
        stats_dicts.append(stats)
        print(stats)

    #print(params['cluster_name'], stats)
    annot_model = BasicEnsemble(models, stats_dicts)
    print(annot_model.stats)

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

def validate_cv_model_noretrain(annot_model, val_path, go_labels, features, baseline_values):
    val_df = pl.read_parquet(val_path)
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
    val_path = node['node']['val_path']
    baseline_values = node['node']['baseline_metrics']
    go_labels = node['node']['go']
    
    results, validation_solved_df = validate_cv_model_noretrain(
        annot_model, val_path, go_labels, features,
        baseline_values=baseline_values)
    
    return annot_model, results, validation_solved_df