import json
#from multiprocessing import Pool
#from mealpy import FloatVar
from multiprocessing import Pool
import random

import numpy as np

#param_sets = json.load(open('config/base_param_bounds.json', 'r'))
#default_params = param_sets[-1]

#'ankh_base', 'ankh_large', 
#'esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 
#'prottrans'

param_bounds = json.load(open('config/base_param_bounds.json', 'r'))

network_mandatory_params = ['l1_dim', 'l2_dim', 'dropout_rate', 'leakyrelu_1_alpha']

param_types = {
    'l1_dim': int,
    'l2_dim': int,
    'leakyrelu_1_alpha': float,
    'dropout_rate': float,
    'final_dim': int,
    'patience': int,
    'epochs': int,
    'learning_rate': float,
    'batch_size': int
}

class ProblemTranslator:
    def __init__(self, my_param_bounds) -> None:
        self.params_list = []
        self.upper_bounds = []
        self.lower_bounds = []
        param_groups = sorted(my_param_bounds.keys())
        for key in param_groups:
            param_names = sorted(my_param_bounds[key].keys())
            for name in param_names:
                lower = float(my_param_bounds[key][name][0])
                upper = float(my_param_bounds[key][name][1])
                self.lower_bounds.append(lower)
                self.upper_bounds.append(upper)
                self.params_list.append((key, name))
        
        '''print('Param bounds:')
        for i in range(len(self.params_list)):
            print(self.params_list[i], self.upper_bounds[i], self.lower_bounds[i])'''

    '''def to_bounds(self):
        return FloatVar(lb=self.lower_bounds, ub=self.upper_bounds)'''
    
    def to_dict(self):
        return {
            'params_list': self.params_list,
            'upper_bounds': self.upper_bounds,
            'lower_bounds': self.lower_bounds,
        }

    def decode(self, vec):
        new_param_dict = {}
        for first, second in self.params_list:
            new_param_dict[first] = {}
        for i in range(len(self.params_list)):
            first, second = self.params_list[i]
            converter = param_types[second]
            val = converter(vec[i])
            new_param_dict[first][second] = val
        
        for param_group_name, params in new_param_dict.items():
            if param_group_name.startswith('esm') or param_group_name.startswith('taxa'):
                for param_name in network_mandatory_params:
                    if not param_name in params:
                        print(new_param_dict)
                        print('Cannot find', param_name, 'in', params)
                        raise Exception("Could not find " + param_name)
        
        return new_param_dict

    def encode(self, param_dict):
        vec = []
        for i in range(len(self.params_list)):
            first, second = self.params_list[i]
            original = param_dict[first][second]
            val = float(original)
            vec.append(val)

        #print('Encoded solution', param_dict, 'to')
        #print(vec)
        return vec

#PARAM_TRANSLATOR = ProblemTranslator()

class RandomSearchMetaheuristic:
    def __init__(self, test_name, param_translator: ProblemTranslator, 
                 pop_size, n_jobs = 24, metric_name=None) -> None:
        self.param_translator = param_translator
        upper_bounds = self.param_translator.upper_bounds
        lower_bounds = self.param_translator.lower_bounds
        assert len(upper_bounds) == len(lower_bounds)
        self.bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(upper_bounds))]
        self.test_name = test_name
        self.pop_size = pop_size
        self.n_jobs = n_jobs
        self.to_optimize = metric_name
        random.seed(1337)
        self.generate_population()
    
    def generate_population(self):
        self.population = []
        #for param_dict in param_sets:
        #    encoded = PARAM_TRANSLATOR.encode(param_dict)
        #    self.population.append(encoded)
        
        for _ in range(self.pop_size):
            new_solution = []
            for lb, ub in self.bounds:
                val = random.uniform(lb, ub)
                new_solution.append(val)
            self.population.append(new_solution)
        
    def new_population(self, top_best: list):
        new_bounds = []
        for i in range(len(self.param_translator.params_list)):
            param_name = self.param_translator.params_list[i]
            param_values = [s[i] for s, f in top_best]
            lb, ub = self.bounds[i]
            new_lb = max(lb, min(param_values))
            new_up = min(ub, max(param_values))
            new_bounds.append((new_lb, new_up))
            print(param_name, ' new min/max is ', new_bounds[-1])
        
        new_pop = []
        for _ in range(self.pop_size):
            new_solution = []
            for lb, ub in new_bounds:
                val = random.uniform(lb, ub)
                new_solution.append(val)
            new_pop.append(new_solution)
        return new_pop
    
    def sort_solutions(solutions):
        #First: Mean fitness, Second: Min fitness, Third: smaller standard deviation
        #solutions.sort(key = lambda tp: (round(tp[1][0], 3), round(tp[1][1], 3), -tp[1][2]))
        solutions.sort(key = lambda tp: tp[1])
    
    def run_tests(self, objective_func, gens=4, top_perc = 0.33, log_dir="/tmp/"):
        all_solutions = []
        report = []

        for gen in range(gens):
            gen_file = log_dir + '/gen_'+str(gen)+'_population.json'
            print(self.test_name, 'gen', gen)

            if self.n_jobs <= 1:
                fitness_vec = [objective_func(a) for a in self.population]
            else:
                with Pool(self.n_jobs) as pool:
                    fitness_vec = pool.map(objective_func, self.population)
            
            log_dict = {
                'population': []
            }
            for sol, m_dict in zip(self.population, fitness_vec):
                log_dict['population'].append({
                    'solution': sol,
                    'metrics': m_dict
                })
            json.dump(log_dict, open(gen_file, 'w'), indent=4)

            fitness_vec = [m_dict[self.to_optimize] for m_dict in fitness_vec]

            solutions_with_fitness = [(self.population[i], fitness_vec[i])
                for i in range(self.pop_size)]
            n_top = int(self.pop_size * top_perc)
            all_solutions += solutions_with_fitness
            RandomSearchMetaheuristic.sort_solutions(all_solutions)
            top_best = all_solutions[-n_top:]
            top_msg = '\nTop ' + str(n_top) + ' solutions at gen ' + str(gen) + ':'
            report.append(top_msg)
            #print(top_msg)
            for s, f in top_best:
                #print('Mean ROC AUC: ' + str(f))
                report.append('Mean ROC AUC: ' + str(f))
            if gen < gens:
                self.population = self.new_population(top_best)

        n_top = int(len(self.population)*top_perc)
        if n_top > 12:
            n_top = 12
        top_best = all_solutions[-n_top:]
        best_solution, best_fitness = all_solutions[-1]
        report.append('Top ' + str(n_top) + ' solutions:')
        for s, f in top_best:
            solution_str = json.dumps(self.param_translator.decode(s), indent=4)
            report += solution_str.split('\n')
            report.append('ROC AUC: ' + str(f))
        
        for i in range(len(self.param_translator.params_list)):
            param_name = self.param_translator.params_list[i]
            param_values = [s[i] for s, f in top_best]
            std = np.std(param_values)
            mean = np.mean(param_values)
            report.append(str(param_name) + ' mean: ' + str(mean) + '; std:' + str(std))
        report = '\n'.join(report)
        #print(report)
        return best_solution, best_fitness, report



