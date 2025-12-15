from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
print('New thread', file=sys.stderr)

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from mf_swarm_lib.core.metaheuristics import ProblemTranslator, GeneticAlgorithm, RandomSearchMetaheuristic
from mf_swarm_lib.data.dataset import Dataset, find_latest_dataset
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.core.node_factory import create_params_for_features, sample_train_test
from mf_swarm_lib.utils.util_base import run_command
from base_benchmark import BaseBenchmarkRunner, run_validation

def run_taxa_test(exp):
    print('Preparing', exp['name'], exp['features'])
    name = exp['name']
    features = exp['features']
    nodes = exp['nodes']
    node_name = list(nodes.keys())[0]
    node = nodes[node_name]
    custom_param_bounds = exp['param_bounds']

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
    params_dict_custom = create_params_for_features(features, 
        bounds=custom_param_bounds, convert_plm_dims=False)
    #del params_dict_custom['taxa']
    #del params_dict_custom['taxa_profile']
    json.dump(params_dict_custom, 
        open(local_dir + '/params_dict_custom_raw.json', 'w'), 
        indent=4)
    problem_translator = ProblemTranslator(params_dict_custom)
    json.dump(problem_translator.to_dict(), 
        open(local_dir + '/params_dict_custom.json', 'w'), 
        indent=4)
    #meta_test = MetaheuristicTest(name, params_list, features, 11)
    
    heuristic_model = RandomSearchMetaheuristic(name, problem_translator, 160,
        n_jobs=6, metric_name="fitness", metric_name2 = 'f1_score_w_06')
    runner = BaseBenchmarkRunner(problem_translator, params_dict, features)
    print('Running', exp['name'])
    best_solution, fitness, report = heuristic_model.run_tests(
        runner.objective_func, gens=5, top_perc=0.6, log_dir=local_dir)
    solution_dict = problem_translator.decode(best_solution)
    print('Saving', exp['name'])
    meta_report_path = local_dir + '/optimization.txt'
    open(meta_report_path, 'w').write(report)
    json.dump(solution_dict, open(local_dir + '/solution.json', 'w'), indent=4)

    val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
    validation_solved_df.write_parquet(local_dir+'/validation.parquet')
    return val_results

