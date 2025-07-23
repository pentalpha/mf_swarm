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
            print(' '.join(cmd))
            run_command(cmd)
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
    dataset = find_or_create_dataset(datasets_dir, dataset_type, min_proteins_per_mf, 
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