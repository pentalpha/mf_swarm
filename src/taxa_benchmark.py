import sys
import os

# Add the directory containing mf_swarm_lib to the python path (src/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
from mf_swarm_lib.core.metaheuristics import ProblemTranslator, GeneticAlgorithm, RandomSearchMetaheuristic
from mf_swarm_lib.data.dataset import Dataset, find_latest_dataset
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.core.node_factory import create_params_for_features, sample_train_test
from mf_swarm_lib.utils.util_base import run_command
from base_benchmark import BaseBenchmarkRunner, run_validation
from mf_swarm_lib.benchmarks.taxa_benchmark import *

if __name__ == '__main__':
    dimension_db_releases_dir = sys.argv[1]
    #hint: 1
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #hint: 30
    min_proteins_per_mf    = int(sys.argv[4])
    val_perc    = float(sys.argv[5])
    test_perc    = float(sys.argv[6])
    pair_benchmark_dir = sys.argv[7]
    taxon_benchmark_path = sys.argv[8]
    encoder_name = sys.argv[9]
    local_dir    = taxon_benchmark_path + '/' + encoder_name
    print(sys.argv)

    run_command(['mkdir -p', local_dir])

    dataset_type           = 'taxon_benchmark'
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
    if encoder_name != 'None':
        top_feature_list = [encoder_name, 'ankh_base', 'prottrans']
    else:
        top_feature_list = ['ankh_base', 'prottrans']
    #get top feature list
    #add current taxa encoder to feature list
    custom_bounds_path = pair_benchmark_dir + '/good_param_bounds.json'
    bounds_dict = json.load(open(custom_bounds_path, 'r'))
    bounds_dict["taxa"]={
        "l1_dim": [64, 512],
        "l2_dim": [64, 512],
        "dropout_rate": [0.2, 0.5],
        "leakyrelu_1_alpha": [0.03, 0.9]
    }
    bounds_dict["taxa_profile"]={
        "l1_dim": [64, 512],
        "l2_dim": [64, 512],
        "dropout_rate": [0.2, 0.5],
        "leakyrelu_1_alpha": [0.03, 0.9]
    }
    name = path.basename(local_dir)
    experiment = {
        'feature_combo': name,
        'name': dataset_type+'_'+name,
        'features': top_feature_list,
        'nodes': dataset.go_clusters,
        'test_perc': test_perc,
        'local_dir': local_dir,
        'param_bounds': bounds_dict
    }

    result_path = path.dirname(local_dir) + '/'+experiment['feature_combo']+'.json'

    if not path.exists(result_path):
        result = run_taxa_test(experiment)
        json.dump(result, open(result_path, 'w'), indent=4)
    else:
        result = json.load(open(result_path, 'r'))

