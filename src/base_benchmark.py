from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
print('New thread', file=sys.stderr)

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import keras
from tqdm import tqdm
import polars as pl
from sklearn import metrics
import numpy as np

from metaheuristics import ProblemTranslator, RandomSearchMetaheuristic, param_bounds
from create_dataset import Dataset, find_latest_dataset
from dimension_db import DimensionDB
from node_factory import create_params_for_features, sample_train_test, train_node
from util_base import run_command, plm_sizes

class BaseBenchmarkRunner():
    def __init__(self, param_translator, node, features):
        self.new_param_translator = param_translator
        self.node = node
        self.features = features
        
    def objective_func(self, solution):
        #print('objective_func', file=sys.stderr)
        new_params_dict = self.new_param_translator.decode(solution)
        self.node['params_dict'] = new_params_dict
        #print('getting roc_auc s', file=sys.stderr)
        model_obj, metrics = train_node(self.node, self.features)
        print(metrics)
        #print(roc_aucs, file=sys.stderr)
        #print('objective_func finish', file=sys.stderr)
        return metrics

def run_validation(node, solution_dict, features):
    node['params_dict'] = solution_dict
    annot_model, stats = train_node(node, features)
    print('Validating')
    val_path = node['node']['val_path']
    go_labels = node['node']['go']
    val_df = pl.read_parquet(val_path)
    val_x_np = []
    for col in features:
        val_x_np.append(val_df[col].to_numpy())
    val_df_y_np = val_df['labels'].to_numpy()
    val_y_pred = annot_model.predict(val_x_np, verbose=0)
    acc = np.mean(keras.metrics.binary_accuracy(val_df_y_np, val_y_pred).numpy())
    roc_auc_score_mac = metrics.roc_auc_score(val_df_y_np, val_y_pred, average='macro')
    roc_auc_score_w = metrics.roc_auc_score(val_df_y_np, val_y_pred, average='weighted')
    print(roc_auc_score_w)
    y_pred_04 = (val_y_pred > 0.4).astype(int)
    y_pred_05 = (val_y_pred > 0.5).astype(int)
    y_pred_06 = (val_y_pred > 0.6).astype(int)
    f1_score = metrics.f1_score(val_df_y_np, y_pred_05, average='macro')
    f1_score_w_05 = metrics.f1_score(val_df_y_np, y_pred_05, average='weighted')
    f1_score_w_06 = metrics.f1_score(val_df_y_np, y_pred_06, average='weighted')
    recall_score = metrics.recall_score(val_df_y_np, y_pred_05, average='macro')
    recall_score_w_05 = metrics.recall_score(val_df_y_np, y_pred_05, average='weighted')
    recall_score_w_06 = metrics.recall_score(val_df_y_np, y_pred_06, average='weighted')
    precision_score = metrics.precision_score(val_df_y_np, y_pred_05, average='macro')
    precision_score_w_05 = metrics.precision_score(val_df_y_np, y_pred_05, average='weighted')
    precision_score_w_06 = metrics.precision_score(val_df_y_np, y_pred_06, average='weighted')
    
    val_stats = {'ROC AUC': float(roc_auc_score_mac),
        'ROC AUC W': float(roc_auc_score_w),
        'Accuracy': float(acc), 
        'f1_score': f1_score,
        'f1_score_w_05': f1_score_w_05,
        'f1_score_w_06': f1_score_w_06,
        'recall_score': recall_score,
        'recall_score_w_05': recall_score_w_05,
        'recall_score_w_06': recall_score_w_06,
        'precision_score': precision_score,
        'precision_score_w_05': precision_score_w_05,
        'precision_score_w_06': precision_score_w_06,
        'val_x': len(val_df)
    }
    results = {
        'test': stats,
        'validation': val_stats,
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

def run_basebenchmark_test(exp):
    print('Preparing', exp['name'], exp['features'])
    name = exp['name']
    features = exp['features']
    nodes = exp['nodes']
    node_name = list(nodes.keys())[0]
    node = nodes[node_name]

    local_dir = exp['local_dir']
    test_perc = exp['test_perc']

    run_command(['mkdir -p', local_dir])

    print('Separating train and test', exp['name'])
    sample_train_test(node['traintest_path'], features, test_perc)
    params_dict = {
        'test_perc': test_perc,
        'node': node,
        'node_name': node_name
    }

    print('Preparing', exp['name'])
    params_dict_custom = create_params_for_features(features)
    del params_dict_custom['taxa']
    del params_dict_custom['taxa_profile']
    problem_translator = ProblemTranslator(params_dict_custom)
    #meta_test = MetaheuristicTest(name, params_list, features, 11)
    heuristic_model = RandomSearchMetaheuristic(name, problem_translator, 120,
        n_jobs=3, metric_name="ROC AUC W", metric_name2 = 'f1_score_w_06')
    runner = BaseBenchmarkRunner(problem_translator, params_dict, features)
    print('Running', exp['name'])
    best_solution, fitness, report = heuristic_model.run_tests(
        runner.objective_func, gens=4, top_perc=0.5, log_dir=local_dir)
    solution_dict = problem_translator.decode(best_solution)
    print('Saving', exp['name'])
    meta_report_path = local_dir + '/optimization.txt'
    open(meta_report_path, 'w').write(report)
    json.dump(solution_dict, open(local_dir + '/solution.json', 'w'), indent=4)

    val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
    validation_solved_df.write_parquet(local_dir+'/validation.parquet')
    json.dump(problem_translator.to_dict(), open(local_dir + '/params_dict_custom.json', 'w'), indent=4)
    return val_results

if __name__ == '__main__':
    dimension_db_releases_dir = sys.argv[1]
    #hint: 1
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #hint: 30
    min_proteins_per_mf    = int(sys.argv[4])
    val_perc    = float(sys.argv[5])
    test_perc    = float(sys.argv[6])
    base_benchmark_dir    = sys.argv[7]
    feature_to_test    = sys.argv[8]
    print(sys.argv)

    feature_to_test = path.basename(feature_to_test).split('.')[1]
    print(feature_to_test)

    run_command(['mkdir -p', base_benchmark_dir])

    dataset_type           = 'base_benchmark'
    matching_dataset_path = find_latest_dataset(datasets_dir, dataset_type, 
                                            min_proteins_per_mf, dimension_db_release_n,
                                            val_perc)
    if matching_dataset_path != None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=False)
        dataset = Dataset(dimension_db=dimension_db, 
                      min_proteins_per_mf=min_proteins_per_mf, 
                      dataset_type=dataset_type,
                      val_perc=val_perc)
    print('Nodes in dataset:', dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)
    
    experiment = {
        'name': dataset_type+'_'+feature_to_test,
        'features': [feature_to_test],
        'nodes': dataset.go_clusters,
        'test_perc': test_perc,
        'local_dir': base_benchmark_dir + '/' + feature_to_test
    }

    result = run_basebenchmark_test(experiment)
    
    json.dump(result, open(base_benchmark_dir + '/'+feature_to_test+'.json', 'w'), indent=4)