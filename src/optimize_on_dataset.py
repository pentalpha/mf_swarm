import mf_swarm_lib
import mf_swarm_lib
from mf_swarm_lib.training.training_processes import run_standard_training
import sys
import json
import os
from os import path
from pickle import dump
import argparse

from tqdm import tqdm
from mf_swarm_lib.utils.util_base import run_command
from mf_swarm_lib.data.dataset import Dataset

# Add the directory containing mf_swarm_lib to the python path (src/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

print("New thread", file=sys.stderr)
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mf_swarm_lib.core.param_translator import ProblemTranslator
from mf_swarm_lib.utils.util_base import create_optimization_stats_report as extract_stats

def sort_experiments(experiments):
    #experiments.sort(key=lambda x: x['result']['fitness'])
    pass

if __name__ == "__main__":
    print(sys.argv)

    parser = argparse.ArgumentParser(description='Optimize on dataset')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--training-dir', type=str, help='Path to training directory')
    parser.add_argument('--pop-size', type=int, help='Population size')
    parser.add_argument('--n-jobs', type=int, help='Number of jobs')
    parser.add_argument('--ready-solutions', type=str, help='Path to ready solutions')
    parser.add_argument('--param-bounds', type=str, help='Path to param bounds')
    parser.add_argument('--report-path', type=str, help='Path to report')
    args = parser.parse_args()
    
    local_dir = args.training_dir
    report_path = args.report_path
    pop_size = args.pop_size
    n_jobs = args.n_jobs
    ready_solutions_paths = args.ready_solutions
    ready_solutions = [json.load(open(p, 'r')) for p in ready_solutions_paths.split(',')]
    param_bounds_path = args.param_bounds
    custom_param_bounds = json.load(open(param_bounds_path, 'r'))
    gens = 0
    dataset = Dataset(dataset_path=args.dataset_path)

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
        stats = extract_stats(local_dir, standard_trainings)
        experiment = {
            'metaparameters': base_params,
            'result': stats
        }
        experiments.append(experiment)
        n_id += 1

    sort_experiments(experiments)

    worst = experiments[0]
    middle = experiments[len(experiments)//2]
    good = experiments[-3]
    very_good = experiments[-2]
    best = experiments[-1]

    optimization_report_json = {
        'dataset_config': dataset_config,
        'experiments': {
            'all': experiments,
            'worst': worst,
            'middle': middle,
            'good': good,
            'very_good': very_good,
            'best': best
        }
    }
    
    json.dump(optimization_report_json, open(report_path, 'w'), 
        indent=4)