import sys
import json
import os
from os import path
from pickle import dump

print("New thread", file=sys.stderr)
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from cross_validation import CrossValRunner, split_train_test_n_folds, validate_cv_model
from metaheuristics import RandomSearchMetaheuristic
from param_translator import ProblemTranslator

if __name__ == '__main__':
    print(sys.argv)

    params_json_path = sys.argv[1]
    results_json_path = sys.argv[2]
    local_dir = path.dirname(results_json_path)

    params_dict = json.load(open(params_json_path, 'r'))
    n_folds = params_dict['n_folds']
    pop_size = params_dict['pop_size']
    n_jobs = params_dict['n_jobs']
    gens = params_dict['gens']
    top_perc = params_dict['top_perc']
    node_name = params_dict['node_name']
    node = params_dict['node']
    features = params_dict['features']
    ready_solutions = params_dict.get('ready_solutions', [])
    print(features)
    #custom_param_bounds = params_dict['param_bounds']
    #print(custom_param_bounds)
    print(node_name)

    problem_translator = ProblemTranslator(None, raw_values=params_dict['problem_translator'])
    
    split_train_test_n_folds(node['traintest_path'], features)
    heuristic_model = RandomSearchMetaheuristic(node_name, problem_translator, pop_size,
        n_jobs=n_jobs, metric_name="fitness", metric_name2 = 'f1_score_w_06', 
        ready_solutions=ready_solutions)
    
    runner = CrossValRunner(problem_translator, params_dict, features, n_folds)
    print('Running', node_name)
    best_solution, fitness, report = heuristic_model.run_tests(
        runner.objective_func, gens=gens, top_perc=top_perc, log_dir=local_dir)
    solution_dict = problem_translator.decode(best_solution)
    print('Saving', node_name)
    meta_report_path = local_dir + '/optimization.txt'
    open(meta_report_path, 'w').write(report)
    json.dump(solution_dict, open(meta_report_path.replace('.txt', '.json'), 'w'), indent=4)

    #val_results, validation_solved_df = run_validation(params_dict, solution_dict, features)
    final_model_ensemble, val_results, validation_solved_df = validate_cv_model(
        params_dict, solution_dict, features, n_folds=n_folds)
    validation_solved_df.write_parquet(local_dir+'/optimized_validation.parquet')
    final_model_ensemble.save(local_dir+'/optimized_model')
    json.dump(val_results, open(results_json_path, 'w'), indent=4)
    #dump(final_model_ensemble, open(local_dir+'/optimized_model.obj', 'wb'))