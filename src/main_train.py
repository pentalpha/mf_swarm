from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
from typing import List

from metaheuristics import ProblemTranslator
from create_dataset import Dataset, find_latest_dataset
from dimension_db import DimensionDB
from util_base import proj_dir, run_command, create_params_for_features


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
        local_dir: str, param_bounds: dict):
    
    print('Preparing', name, features)
    custom_param_bounds = param_bounds

    run_command(['mkdir -p', local_dir])
    
    print('Preparing', name)
    params_dict_custom = create_params_for_features(features, 
        bounds=custom_param_bounds, convert_plm_dims=False)
    
    json.dump(params_dict_custom, 
        open(local_dir + '/params_dict_custom_raw.json', 'w'), 
        indent=4)
    problem_translator = ProblemTranslator(params_dict_custom)
    json.dump(problem_translator.to_dict(), 
        open(local_dir + '/params_dict_custom.json', 'w'), 
        indent=4)

    param_dicts = []
    for node_name, node in nodes.items():

        params_dict = {
            'n_folds': 5,
            'max_proteins': 90000,
            'problem_translator': problem_translator.to_dict(),
            'pop_size': 160,
            'n_jobs': 6,
            'metric_name': "fitness",
            'metric_name2': 'f1_score_w_06',
            'gens': 4,
            'top_perc': 0.6,
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

if __name__ == '__main__':
    dimension_db_releases_dir = sys.argv[1]
    #hint: 1
    dimension_db_release_n    = sys.argv[2]
    datasets_dir              = sys.argv[3]
    #hint: 30
    min_proteins_per_mf       = int(sys.argv[4])
    val_perc                  = float(sys.argv[5])
    optimization_dir          = sys.argv[6]
    #model_output_dir          = sys.argv[7]
    print(sys.argv)

    dataset_type = 'full_swarm'
    matching_dataset_path = find_latest_dataset(datasets_dir, dataset_type, 
                                            min_proteins_per_mf, dimension_db_release_n,
                                            val_perc)
    if matching_dataset_path != None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=True)
        dataset = Dataset(dimension_db=dimension_db, 
                      min_proteins_per_mf=min_proteins_per_mf, 
                      dataset_type=dataset_type,
                      val_perc=val_perc)
    print('Nodes in dataset:', dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)
    
    feature_list = ['taxa_256', 'ankh_base', 'esm2_t33']
    #get top feature list
    #add current taxa encoder to feature list
    custom_bounds_path = proj_dir + '/config/base_param_bounds_v2.larger_taxa.json'
    bounds_dict = json.load(open(custom_bounds_path, 'r'))
    
    name = 'mf_swarm-'+'-'.join(feature_list)

    optimized_metaparameters = run_optimization(
        name=dataset_type+'_'+name,
        features=feature_list,
        nodes=dataset.go_clusters,
        local_dir=optimization_dir,
        param_bounds=bounds_dict)