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

from metaheuristics import ProblemTranslator, GeneticAlgorithm, RandomSearchMetaheuristic
from create_dataset import Dataset, find_latest_dataset
from dimension_db import DimensionDB
from node_factory import create_params_for_features, sample_train_test
from util_base import run_command
from base_benchmark import BaseBenchmarkRunner, run_validation

def run_pair_test(exp):
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
    json.dump(problem_translator.to_dict(), open(local_dir + '/params_dict_custom.json', 'w'), indent=4)
    #meta_test = MetaheuristicTest(name, params_list, features, 11)
    
    heuristic_model = RandomSearchMetaheuristic(name, problem_translator, 8,
        n_jobs=4, metric_name="f1_score_w_06", metric_name2 = 'precision_score_w_06')
    runner = BaseBenchmarkRunner(problem_translator, params_dict, features)
    print('Running', exp['name'])
    best_solution, fitness, report = heuristic_model.run_tests(
        runner.objective_func, gens=2, top_perc=0.6, log_dir=local_dir)
    solution_dict = problem_translator.decode(best_solution)
    print('Saving', exp['name'])
    meta_report_path = local_dir + '/optimization.txt'
    open(meta_report_path, 'w').write(report)
    json.dump(solution_dict, open(local_dir + '/solution.json', 'w'), indent=4)

    val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
    validation_solved_df.write_parquet(local_dir+'/validation.parquet')
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
    base_benchmark_tsv_path = sys.argv[7]
    pair_benchmark_dir    = sys.argv[8]
    print(sys.argv)

    run_command(['mkdir -p', pair_benchmark_dir])

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
    
    feature_pairs = set()

    n_top = 1
    models_sorted = []
    for rawline in open(base_benchmark_tsv_path, 'r'):
        if not rawline.startswith('model\t'):
            cells = rawline.split('\t')
            models_sorted.append((float(cells[1]), cells[0]))
    models_sorted.sort(reverse=True)
    best_models = models_sorted[:n_top]
    for _, x in best_models:
        for _, y in best_models:
            if y != x:
                first, second = sorted([x, y])
                feature_pairs.add((first, second))
        feature_pairs.add((x, models_sorted[-1][1]))
    print('Pairs to test:')
    for a, b in feature_pairs:
        print(a, b)
    
    experiments = []
    for features_to_test in feature_pairs:
        name = '-'.join(list(features_to_test))
        experiment = {
            'feature_combo': name,
            'name': dataset_type+'_'+name,
            'features': [c for c in features_to_test],
            'nodes': dataset.go_clusters,
            'test_perc': test_perc,
            'local_dir': pair_benchmark_dir + '/' + name
        }
        experiments.append(experiment)

    all_results = {}
    for experiment in experiments:
        result_path = pair_benchmark_dir + '/'+experiment['feature_combo']+'.json'
        if not path.exists(result_path):
            result = run_pair_test(experiment)
            json.dump(result, open(result_path, 'w'), indent=4)
        else:
            result = json.load(open(result_path, 'r'))
        all_results[experiment['feature_combo']] = result
    
    json.dump(all_results, open(pair_benchmark_dir + '/pair_results.json', 'w'), indent=4)
    
    