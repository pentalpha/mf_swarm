import sys
import json
from os import path

from cross_validation import split_train_test_n_folds
from metaheuristics import RandomSearchMetaheuristic

if __name__ == '__main__':
    print(sys.argv)

    params_json_path = sys.argv[1]
    results_json_path = sys.argv[2]
    local_dir = path.dirname(results_json_path)

    params_dict = json.load(open(params_json_path, 'r'))
    node_name = params_dict['node_name']
    node = params_dict['node']
    features = params_dict['features']
    print(features)
    custom_param_bounds = params_dict['param_bounds']
    print(custom_param_bounds)
    print(node_name)
    
    split_train_test_n_folds(node['traintest_path'], features, 
        params_dict['n_folds'], max_proteins=params_dict['max_proteins'])
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
    json.dump(solution_dict, open(results_json_path, 'w'), indent=4)

    val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
    validation_solved_df.write_parquet(local_dir+'/validation.parquet')
    return val_results