import sys
import os
from os import path
print("New thread", file=sys.stderr)
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from typing import List
import json

from tqdm import tqdm

from mf_swarm_lib.core.metaheuristics import ProblemTranslator
from mf_swarm_lib.training.train_single_node import training_process
from mf_swarm_lib.data.dataset import Dataset
from mf_swarm_lib.utils.util_base import run_command, plm_sizes
from mf_swarm_lib.utils.extract_optimization_context import (extract_info_from_swarm, 
    optimization_report)
from mf_swarm_lib.core.swarm import Swarm



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
        local_dir: str, param_bounds: dict, ready_solutions: List[dict] = None,
        pop_size = 160, n_jobs = 6, gens = 1, top_perc = 0.6):
    
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
            'pop_size': pop_size,
            'n_jobs': n_jobs,
            'metric_name': "fitness",
            'metric_name2': 'f1_score_w_06',
            'gens': gens,
            'top_perc': top_perc,
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
        local_dir: str, meta_parameters: dict, n_jobs: int, is_test=False):
    
    print('Preparing', name, features)

    run_command(['mkdir -p', local_dir])
    
    

    cafatrain_features = [f.replace('.train', '') for f in features if '.train' in f]
    for feature in cafatrain_features:
        if feature in meta_parameters:
            meta_parameters[feature+'.train'] = meta_parameters[feature]
            del meta_parameters[feature]
        #if feature in meta_parameters['input_dims']:
        #    meta_parameters['input_dims'][feature+'.train'] = meta_parameters['input_dims'][feature]
        #    del meta_parameters['input_dims'][feature]
    meta_parameters['input_dims'] = {f: plm_sizes[f] if f in plm_sizes 
        else plm_sizes[f.replace('.train', '')] 
        for f in features}
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
            'n_jobs': n_jobs,
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
        #cmd = ['python', 'src/train_single_node.py', exp_path, run_result]
        if not path.exists(run_result):
            run_command(['mkdir -p', path.dirname(run_result)])
            training_process(exp_path, run_result, is_test=is_test)
            #print(' '.join(cmd))
            #run_command(cmd)
        if not path.exists(run_result):
            print('Error: Standard training result not found for', node_name)
            quit(1)
        
        trainings_done[node_name] = json.load(open(run_result, 'r'))

    return trainings_done

def single_gen_optimizer(release_dir, local_dir, pop_size, n_jobs, ready_solutions_paths, 
        param_bounds_path, dataset_path, commentary, report_path):
    ready_solutions = [json.load(open(p, 'r')) 
        for p in ready_solutions_paths.split(',')]
    custom_param_bounds = json.load(open(param_bounds_path, 'r'))
    gens = 0
    dataset = Dataset(dataset_path=dataset_path)
    dataset_config = {
        'dataset_type': dataset.dataset_type,
        'min_proteins_per_mf': dataset.min_proteins_per_mf,
        'val_perc': dataset.val_perc
    }

    problem_translator = ProblemTranslator(custom_param_bounds)

    population_raw = problem_translator.generate_population(pop_size, ready_solutions)
    population = [problem_translator.decode(sol) for sol in population_raw]
    feature_list = dataset.datasets_to_load

    n_id = 1
    experiments = []
    for base_params in tqdm(population):
        run_command(['rm -rf', local_dir])
        name = f'test_{dataset.dataset_type}_{n_id}'
        standard_trainings = run_standard_training(name, feature_list, dataset.go_clusters,
            local_dir, base_params, n_jobs=n_jobs, is_test=True)
        plots_dir = local_dir+'/plots'
        run_command(['mkdir -p', plots_dir])
        #draw_swarm_panel(local_dir, plots_dir)
        json.dump(dataset_config, open(local_dir+'/configs_used.json', 'w'))
        new_swarm = Swarm(local_dir, release_dir+'/go-basic.obo')
        stats = extract_info_from_swarm(local_dir)
        experiment = {
            'metaparameters': base_params,
            'result': stats
        }
        experiments.append(experiment)
        n_id += 1

    report_dict = optimization_report(
        commentary = commentary,
        experiments = experiments,
        bounds_dict = custom_param_bounds,
        dataset_config = dataset_config
    )
    
    json.dump(report_dict, open(report_path, 'w'), 
        indent=4)
    
    return report_dict