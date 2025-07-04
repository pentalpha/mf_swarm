import sys
import json
from os import path

from base_benchmark import run_validation
from cross_validation import CrossValRunner, split_train_test_n_folds
from metaheuristics import RandomSearchMetaheuristic
from param_translator import ProblemTranslator

N_FOLDS = 5

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
    #custom_param_bounds = params_dict['param_bounds']
    #print(custom_param_bounds)
    print(node_name)

    problem_translator = ProblemTranslator(None, raw_values=params_dict['problem_translator'])
    
    split_train_test_n_folds(node['traintest_path'], features)
    heuristic_model = RandomSearchMetaheuristic(node_name, problem_translator, 160,
        n_jobs=5, metric_name="fitness", metric_name2 = 'f1_score_w_06')
    
    runner = CrossValRunner(problem_translator, params_dict, features, N_FOLDS)
    print('Running', node_name)
    best_solution, fitness, report = heuristic_model.run_tests(
        runner.objective_func, gens=5, top_perc=0.6, log_dir=local_dir)
    solution_dict = problem_translator.decode(best_solution)
    print('Saving', node_name)
    meta_report_path = local_dir + '/optimization.txt'
    open(meta_report_path, 'w').write(report)
    json.dump(solution_dict, open(results_json_path, 'w'), indent=4)

    val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
    validation_solved_df.write_parquet(local_dir+'/validation.parquet')
    json.dump(val_results, open(results_json_path, 'w'), indent=4)