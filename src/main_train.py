from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
from typing import List

from metaheuristics import ProblemTranslator
from create_dataset import Dataset, find_latest_dataset, find_or_create_dataset
from dimension_db import DimensionDB
from plotting import draw_swarm_panel
from swarm import Swarm
from train_single_node import training_process
from util_base import proj_dir, run_command, create_params_for_features, general_configs


'''
split_train_test_n_folds(node['traintest_path'], features, 5, max_proteins=90000)
heuristic_model = RandomSearchMetaheuristic(name, problem_translator, 160,
    n_jobs=6, metric_name="fitness", metric_name2 = 'f1_score_w_06')
runner = BaseBenchmarkRunner(problem_translator, params_dict, features)
print('Running', name)
best_solution, fitness, report = heuristic_model.run_tests(
    runner.objective_func, gens=5, top_perc=0.6, log_dir=local_dir)
solution_dict = problem_translator.decode(best_solution)
print('Saving', name)
meta_report_path = local_dir + '/optimization.txt'
open(meta_report_path, 'w').write(report)
json.dump(solution_dict, open(local_dir + '/solution.json', 'w'), indent=4)

val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
validation_solved_df.write_parquet(local_dir+'/validation.parquet')
return val_results
'''

def run_optimization(name: str, features: List[str], nodes: dict,
        local_dir: str, param_bounds: dict, ready_solutions: List[dict] = None):
    
    print('Preparing', name, features)

    run_command(['mkdir -p', local_dir])
    
    print('Preparing', name)
    json.dump(param_bounds, 
        open(local_dir + '/param_bounds_raw.json', 'w'), 
        indent=4)
    problem_translator = ProblemTranslator(param_bounds)
    json.dump(problem_translator.to_dict(), 
        open(local_dir + '/param_bounds_parsed.json', 'w'), 
        indent=4)

    param_dicts = []
    for node_name, node in nodes.items():

        params_dict = {
            'n_folds': 5,
            'max_proteins': 90000,
            'problem_translator': problem_translator.to_dict(),
            'ready_solutions': ready_solutions if ready_solutions else [],
            'pop_size': general_configs['pop_size'],
            'n_jobs': general_configs['n_jobs'],
            'metric_name': "fitness",
            'metric_name2': 'f1_score_w_06',
            'gens': general_configs['gens'],
            'top_perc': general_configs['top_perc'],
            'log_dir': local_dir + '/logs/' + node_name,
            'features': features,
            'node': node,
            'node_name': node_name
        }
        param_dicts.append(params_dict)
    param_dicts.sort(key=lambda node: len(node['node']['id']))

    node_experiments = []
    for params_dict in param_dicts:
        node_dir = local_dir + '/' + params_dict['node_name']
        run_command(['mkdir -p', node_dir])
        exp_params_path = node_dir + '/exp_params.json'
        json.dump(params_dict, open(exp_params_path, 'w'), indent=4)
        node_experiments.append(exp_params_path)
    
    print('Node experiments sequence:')
    for x in node_experiments:
        print(x)

    optimizations_done = {}
    for exp_path in node_experiments:
        node_name = path.basename(path.dirname(exp_path))
        exp_result = exp_path.replace('params', 'results')
        cmd = ['python', 'src/optimize_single_node.py', exp_path, exp_result]
        if not path.exists(exp_result):
            run_command(['mkdir -p', path.dirname(exp_result)])
            print(' '.join(cmd))
            run_command(cmd)
        if not path.exists(exp_result):
            print('Error: Optimization result not found for', node_name)
            quit(1)
        
        optimizations_done[node_name] = json.load(open(exp_result, 'r'))

    return optimizations_done

def run_standard_training(name: str, features: List[str], nodes: dict,
        local_dir: str, meta_parameters: dict):
    
    print('Preparing', name, features)

    run_command(['mkdir -p', local_dir])
    
    print('Preparing', name)
    json.dump(meta_parameters, 
        open(local_dir + '/standard_params.json', 'w'), 
        indent=4)

    param_dicts = []
    for node_name, node in nodes.items():
        params_dict = {
            'n_folds': 5,
            'max_proteins': 90000,
            'params_dict': meta_parameters,
            'n_jobs': general_configs['n_jobs'],
            'log_dir': local_dir + '/logs/' + node_name,
            'features': features,
            'node': node,
            'node_name': node_name
        }
        param_dicts.append(params_dict)
    param_dicts.sort(key=lambda node: len(node['node']['id']))

    node_experiments = []
    for params_dict in param_dicts:
        node_dir = local_dir + '/' + params_dict['node_name']
        run_command(['mkdir -p', node_dir])
        exp_params_path = node_dir + '/standard_params.json'
        json.dump(params_dict, open(exp_params_path, 'w'), indent=4)
        node_experiments.append(exp_params_path)
    
    print('Node run training sequence:')
    for x in node_experiments:
        print(x)

    trainings_done = {}
    for exp_path in node_experiments:
        node_name = path.basename(path.dirname(exp_path))
        run_result = exp_path.replace('params', 'results')
        cmd = ['python', 'src/train_single_node.py', exp_path, run_result]
        if not path.exists(run_result):
            run_command(['mkdir -p', path.dirname(run_result)])
            training_process(exp_path, run_result)
            #print(' '.join(cmd))
            #run_command(cmd)
        if not path.exists(run_result):
            print('Error: Standard training result not found for', node_name)
            quit(1)
        
        trainings_done[node_name] = json.load(open(run_result, 'r'))

    return trainings_done

if __name__ == '__main__':

    dimension_db_releases_dir = general_configs['dimension_db_releases_dir']
    dimension_db_release_n    = general_configs['dimension_db_release_n']
    datasets_dir              = general_configs['datasets_dir']
    min_proteins_per_mf       = int(general_configs['min_proteins_per_mf'])
    val_perc                  = float(general_configs['val_perc'])
    optimization_dir          = general_configs['optimization_dir']
    n_to_optimize          = general_configs['n_to_optimize']
    experiment_name           = sys.argv[1]
    local_dir = path.join(optimization_dir, experiment_name)
    print(sys.argv)

    dataset_type = 'full_swarm'
    dataset, dimension_db = find_or_create_dataset(datasets_dir, dataset_type, min_proteins_per_mf, 
        dimension_db_release_n, dimension_db_releases_dir, val_perc)
    
    feature_list = ['taxa_256', 'ankh_base', 'esm2_t33']
    custom_bounds_path = proj_dir + '/config/base_param_bounds.v2.benchmarked.json'
    bounds_dict = json.load(open(custom_bounds_path, 'r'))
    base_params_path = proj_dir + '/config/base_params_v1.json'
    base_params = json.load(open(base_params_path, 'r'))
    
    name = 'mf_swarm-'+'-'.join(feature_list)

    if not path.exists(local_dir):
        run_command(['mkdir -p', local_dir])
    json.dump(general_configs, open(local_dir + '/configs_used.json', 'w'), indent=4)

    standard_trainings = run_standard_training(name, feature_list, dataset.go_clusters,
        local_dir, base_params)
    #sort standard trainings by AUPRC W
    standard_trainings_sorted = sorted(standard_trainings.keys(), 
        key=lambda node_name: standard_trainings[node_name]['validation']['AUPRC W'])
    if n_to_optimize > 0:
        worst_trainings = standard_trainings_sorted[:n_to_optimize]

        clusters_subset = {}
        print('Worst trainings to optimize:')
        for node_name in worst_trainings:
            print(node_name, standard_trainings[node_name]['validation']['AUPRC W'])
            clusters_subset[node_name] = dataset.go_clusters[node_name]

        optimized_metaparameters = run_optimization(
            name=dataset_type+'_'+name,
            features=feature_list,
            nodes=clusters_subset,
            local_dir=local_dir,
            param_bounds=bounds_dict,
            ready_solutions=[base_params])
    
    plots_dir = local_dir+'/plots'
    run_command(['mkdir -p', plots_dir])
    draw_swarm_panel(local_dir, plots_dir)

    new_swarm = Swarm(local_dir, dimension_db.go_basic_path)
    #df, go_final_seq, val_df, val_go_final_seq = new_swarm.make_all_predictions(dimension_db)
    
    '''
    import numpy as np
    import polars as pl
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import r2_score
    from sklearn import metrics
    from tqdm import tqdm

#%%

n_proteins = 15000
scores_per_line = 120

raw_df = pl.read_parquet(new_swarm.raw_scores_path)
scores_matrix = raw_df['scores'].to_numpy()
labels_matrix = raw_df['labels'].to_numpy()
protein_indexes = np.random.choice(scores_matrix.shape[0], size=n_proteins, replace=False)
scores_matrix = scores_matrix[protein_indexes]

print('Creating features and labels')
feature_lines = []
feature_names = ['dist_to_root', 'current_score', 'top_min', 'top_q1', 'top_q2', 'top_q3', 'top_max',
    'bottom_min', 'bottom_q1', 'bottom_q2', 'bottom_q3', 'bottom_max']
correct_scores = []
for i in tqdm(range(scores_matrix.shape[0])):
    prot_scores = scores_matrix[i]
    correct_preds = labels_matrix[i]
    positive_indexes = [go_id_index for go_id_index, val in enumerate(correct_preds) if val == 1]
    max_positives = int(scores_per_line / 2)
    if len(positive_indexes) > max_positives:
        #print(positive_indexes)
        #print(max_positives)
        positive_indexes2 = list(np.random.choice(positive_indexes, size=max_positives, replace=False))
    else:
        positive_indexes2 = positive_indexes
    positive_indexes = positive_indexes2
    n_negatives = len(positive_indexes)*2
    neg_indexes = [go_id_index for go_id_index, val in enumerate(correct_preds) if val == 0]
    neg_indexes = list(np.random.choice(neg_indexes, size=n_negatives, replace=False))
    all_indexes = positive_indexes + neg_indexes
    
    for go_id_index in all_indexes:
        go_id = new_swarm.go_id_sequence[go_id_index]
        top_indexes = new_swarm.go_descendants_indexes[go_id]
        bottom_indexes = new_swarm.go_ancestors_indexes[go_id]
        top_scores = prot_scores[top_indexes]
        bottom_scores = prot_scores[bottom_indexes]
        if len(top_scores) > 0:
            top_features = [np.min(top_scores), np.quantile(top_scores, 0.25), np.quantile(top_scores, 0.5), np.quantile(top_scores, 0.75), np.max(top_scores)]
        else:
            top_features = [0, 0, 0, 0, 0]
        if len(bottom_scores) > 0:
            bottom_features = [np.min(bottom_scores), np.quantile(bottom_scores, 0.25), np.quantile(bottom_scores, 0.5), np.quantile(bottom_scores, 0.75), np.max(bottom_scores)]
        else:
            bottom_features = [0, 0, 0, 0, 0]
        feature_line = [new_swarm.go_distances_to_root[go_id], prot_scores[go_id_index]] + top_features + bottom_features
        feature_lines.append(np.array(feature_line))
        correct_scores.append(correct_preds[go_id_index])

feature_lines = np.asarray(feature_lines)
correct_scores = np.asarray(correct_scores)
print('Features matrix shape:', feature_lines.shape)
print('Labels matrix shape:', correct_scores.shape)
df_path = 'tmp/final_node_df.parquet'
df = pl.DataFrame({'features': feature_lines,
    'labels': correct_scores})
df.write_parquet(df_path)
#%%
print('Split data and train RandomForestRegressor')
df = pl.read_parquet(df_path)
feature_lines = df['features'].to_numpy()
correct_scores = df['labels'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(feature_lines, correct_scores, test_size=0.2, random_state=42)
pred_scores = X_test[:, 1]
#%%
for a in range(100):
    pred_bool = pred_scores > a/100
    f1 = metrics.f1_score(y_test, pred_bool)
    print('Base f1:', f1)
#%%
rf = RandomForestClassifier(n_estimators=100, random_state=1337, n_jobs=30)
print('Fitting')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Test R^2:', r2_score(y_test, y_pred))
    '''