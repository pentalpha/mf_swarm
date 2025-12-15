
import os
from os import path
import polars as pl
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import keras
from sklearn import metrics

from mf_swarm_lib.core.metaheuristics import ProblemTranslator, RandomSearchMetaheuristic
from mf_swarm_lib.utils.util_base import run_command, plm_sizes
from mf_swarm_lib.core.node_factory import create_params_for_features, train_node

class MetaheuristicTest():

    def __init__(self, name, params, features, pop) -> None:
        self.nodes = params

        '''self.problem_constrained = {
            "obj_func": self.objective_func,
            "bounds": PARAM_TRANSLATOR.to_bounds(),
            "minmax": "max",
            "log_to": "file",
            "log_file": "result.log",         # Default value = "mealpy.log"
        }'''

        params_dict = create_params_for_features(features)
        self.features = features
        self.new_param_translator = ProblemTranslator(params_dict)
        self.heuristic_model = RandomSearchMetaheuristic(name, self.new_param_translator, pop,
            n_jobs=3)
    
    def objective_func(self, solution):
        #print('objective_func', file=sys.stderr)
        new_params_dict = self.new_param_translator.decode(solution)
        
        for node in self.nodes:
            node['params_dict'] = new_params_dict
        '''n_procs = config['training_processes']
        if n_procs > len(self.nodes):
            n_procs = len(self.nodes)'''

        #print('getting roc_auc s', file=sys.stderr)
        roc_aucs = [train_node(node, self.features)[1]['ROC AUC'] for node in self.nodes]
        #print(roc_aucs, file=sys.stderr)
        mean_training_roc = np.mean(roc_aucs)
        min_roc_auc = min(roc_aucs)
        std = np.std(roc_aucs)
        #print('objective_func finish', file=sys.stderr)
        return mean_training_roc, min_roc_auc, std

    def test(self):
        best_solution, best_fitness, report = self.heuristic_model.run_tests(
            self.objective_func, gens=2, top_perc=0.5)
        solution_dict = self.new_param_translator.decode(best_solution)
        #print(json.dumps(solution_dict, indent=4))
        print(best_fitness)

        results = {}
        rocs = []
        for node in self.nodes:
            node['params_dict'] = solution_dict
            annot_model, stats = train_node(node, self.features)
            print('Validating')
            val_path = node['node']['val_path']
            val_df = pl.read_parquet(val_path)
            val_x_np = []
            for col in self.features:
                val_x_np.append(val_df[col].to_numpy())
            val_df_y_np = val_df['labels'].to_numpy()
            val_y_pred = annot_model.predict(val_x_np, verbose=0)
            roc_auc_score = metrics.roc_auc_score(val_df_y_np, val_y_pred)
            acc = np.mean(keras.metrics.binary_accuracy(val_df_y_np, val_y_pred).numpy())
            print(roc_auc_score)
            results[node['node_name']] = {
                'test': stats,
                'validation': {
                    'roc_auc_score': float(roc_auc_score),
                    'acc': float(acc),
                    'val_x': len(val_df)
                }
            }
            if roc_auc_score == roc_auc_score:
                rocs.append(roc_auc_score)
        results['roc_auc_mean'] = np.mean(rocs)
        results['roc_auc_min'] = min(rocs)
        results['params_dict'] = solution_dict

        return results, best_fitness, report