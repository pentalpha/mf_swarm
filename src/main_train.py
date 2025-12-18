from mf_swarm_lib.data.dimension_db import DimensionDB
from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
from typing import List
import argparse

# Add the directory containing mf_swarm_lib to the python path (src/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mf_swarm_lib.data.dataset import Dataset
from mf_swarm_lib.utils.plotting import draw_swarm_panel
from mf_swarm_lib.core.swarm import Swarm
from mf_swarm_lib.utils.util_base import proj_dir, run_command
from mf_swarm_lib.training.training_processes import run_optimization, run_standard_training

#Use CLI arguments to get the parameters
def parse_args():
    parser = argparse.ArgumentParser(description='Swarm trainer')
    parser.add_argument('--experiment-name', type=str, help='Name of the experiment')
    #dimension_db_releases_dir
    parser.add_argument('--dimension-db-releases-dir', type=str, help='Path to the dimension db releases directory')
    #dimension_db_release_n
    parser.add_argument('--dimension-db-release-n', type=str, help='Dimension DB release number/identifier')
    parser.add_argument('--dataset-dir', type=str, help='Path to the dataset directory')
    #min_proteins_per_mf
    parser.add_argument('--min-proteins-per-mf', type=int, help='Minimum number of proteins per Molecular Function (MF)')
    #val_perc
    parser.add_argument('--val-perc', type=float, help='Validation set percentage (0.0 to 1.0)')
    #base_params_path
    parser.add_argument('--base-params-path', type=str, help='Path to the base parameters file')
    #n_jobs
    parser.add_argument('--n-jobs', type=int, help='Number of jobs to run in parallel')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    experiment_name = args.experiment_name
    dimension_db_releases_dir = args.dimension_db_releases_dir
    dimension_db_release_n = args.dimension_db_release_n
    dataset_dir = args.dataset_dir
    min_proteins_per_mf = args.min_proteins_per_mf
    val_perc = args.val_perc
    base_params_path = args.base_params_path
    n_jobs = args.n_jobs
    
    print(sys.argv)

    dataset = Dataset(dataset_path=dataset_dir)
    dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=False)
    
    feature_list = dataset.datasets_to_load
    base_params = json.load(open(base_params_path, 'r'))
    
    name = 'mf_swarm-'+'-'.join(feature_list)

    local_dir = experiment_name
    if not path.exists(local_dir):
        run_command(['mkdir -p', local_dir])

    standard_trainings = run_standard_training(name, feature_list, dataset.go_clusters,
        local_dir, base_params, n_jobs=n_jobs)
    #sort standard trainings by AUPRC W
    standard_trainings_sorted = sorted(standard_trainings.keys(), 
        key=lambda node_name: standard_trainings[node_name]['validation']['AUPRC W'])
    '''if n_to_optimize > 0:
        worst_trainings = standard_trainings_sorted[:n_to_optimize]

        clusters_subset = {}
        print('Worst trainings to optimize:')
        for node_name in worst_trainings:
            print(node_name, standard_trainings[node_name]['validation']['AUPRC W'])
            clusters_subset[node_name] = dataset.go_clusters[node_name]

        optimized_metaparameters = run_optimization(
            name=dataset.dataset_type+'_'+name,
            features=feature_list,
            nodes=clusters_subset,
            local_dir=local_dir,
            param_bounds=bounds_dict,
            ready_solutions=[base_params])'''
    
    plots_dir = local_dir+'/plots'
    run_command(['mkdir -p', plots_dir])
    draw_swarm_panel(local_dir, plots_dir)

    #new_swarm = Swarm(local_dir, dimension_db.go_basic_path)
