import sys
import os
import argparse

# Add the directory containing mf_swarm_lib to the python path (src/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mf_swarm_lib.training.training_processes import single_gen_optimizer


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
    parser.add_argument('--commentary', type=str, help='Commentary')
    parser.add_argument('--pddb-release-dir', type=str)
    args = parser.parse_args()
    
    local_dir = args.training_dir
    report_path = args.report_path
    pop_size = args.pop_size
    n_jobs = args.n_jobs
    ready_solutions_paths = args.ready_solutions
    param_bounds_path = args.param_bounds
    dataset_path = args.dataset_path
    pddb_release_dir = args.pddb_release_dir
    commentary = args.commentary
    
    single_gen_optimizer(pddb_release_dir, local_dir, pop_size, n_jobs, 
        ready_solutions_paths, param_bounds_path, dataset_path, 
        commentary, report_path)

    