from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
from typing import List

# Add the directory containing mf_swarm_lib to the python path (src/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mf_swarm_lib.core.metaheuristics import ProblemTranslator
from mf_swarm_lib.data.dataset import Dataset, find_latest_dataset, find_or_create_dataset
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.utils.plotting import draw_swarm_panel
from mf_swarm_lib.core.swarm import Swarm
from mf_swarm_lib.training.train_single_node import training_process
from mf_swarm_lib.utils.util_base import proj_dir, run_command, create_params_for_features, general_configs
from mf_swarm_lib.training.training_processes import run_optimization, run_standard_training

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
