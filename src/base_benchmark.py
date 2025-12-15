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
import keras
from tqdm import tqdm
import polars as pl
from sklearn import metrics
import numpy as np
from mf_swarm_lib.core.metaheuristics import ProblemTranslator, RandomSearchMetaheuristic, param_bounds, GeneticAlgorithm
from mf_swarm_lib.data.dataset import Dataset, find_latest_dataset
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.core.node_factory import create_params_for_features, sample_train_test, train_node
from mf_swarm_lib.utils.util_base import run_command
from mf_swarm_lib.benchmarks.base_benchmark import *

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

