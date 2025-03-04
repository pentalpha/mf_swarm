import json
#from multiprocessing import Pool
#from mealpy import FloatVar
from multiprocessing import Pool
import random
import signal
from os import path, mkdir
import numpy as np

from param_translator import ProblemTranslator
from plotting import iterative_gens_draw

#param_sets = json.load(open('config/base_param_bounds.json', 'r'))
#default_params = param_sets[-1]

#'ankh_base', 'ankh_large', 
#'esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 
#'prottrans'

param_bounds = json.load(open('config/base_param_bounds_v2.json', 'r'))


class RandomSearchMetaheuristic:
    def __init__(self, test_name, param_translator: ProblemTranslator, 
                 pop_size, n_jobs = 24, metric_name=None, metric_name2=None) -> None:
        self.param_translator = param_translator
        upper_bounds = self.param_translator.upper_bounds
        lower_bounds = self.param_translator.lower_bounds
        assert len(upper_bounds) == len(lower_bounds)
        self.bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(upper_bounds))]
        self.test_name = test_name
        self.pop_size = pop_size
        self.n_jobs = n_jobs
        self.to_optimize = metric_name
        self.to_optimize2 = metric_name2
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
        solutions.sort(key = lambda tp: (round(tp[1][0], 2), tp[1][1]))
        #solutions.sort(key = lambda tp: tp[1])
    
    def get_fitness(self, objective_func):
        if self.n_jobs <= 1:
            fitness_vec = [objective_func(a) for a in self.population]
        else:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(self.n_jobs)
            signal.signal(signal.SIGINT, original_sigint_handler)
            try:
                res = pool.map_async(objective_func, self.population)
                fitness_vec = res.get()
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                quit(1)
            else:
                #print("Normal termination")
                pool.close()
            pool.join()
        return fitness_vec

    def run_tests(self, objective_func, gens=4, top_perc = 0.33, log_dir="/tmp/"):
        all_solutions = []
        report = []

        super_log_dir = path.dirname(log_dir)
        print('Base benchmark dir is', super_log_dir)
        last_n = 0
        super_log_dir = super_log_dir if path.exists(super_log_dir) else None
        for gen in range(gens):
            gen_file = log_dir + '/gen_'+str(gen)+'_population.json'
            print(self.test_name, 'gen', gen)
            fitness_dicts = self.get_fitness(objective_func)

            '''if self.n_jobs <= 1:
                fitness_vec = [objective_func(a) for a in self.population]
            else:
                with Pool(self.n_jobs) as pool:
                    fitness_vec = pool.map(objective_func, self.population)'''
            
            # Log generation data
            generation_log = {
                'population': [{'solution': self.param_translator.decode(sol), 'metrics': metrics}
                               for sol, metrics in zip(self.population, fitness_dicts)]
            }
            with open(f"{log_dir}/gen_{gen}_population.json", "w") as f:
                json.dump(generation_log, f, indent=4)

            fitness_vec = [(m_dict[self.to_optimize], m_dict[self.to_optimize2]) 
                for m_dict in fitness_dicts]
            
            if super_log_dir:
                try:
                    new_n = iterative_gens_draw(super_log_dir, prev_n_gens=last_n)
                    last_n = new_n
                except Exception as err:
                    print('Exception while trying to draw evolution:')
                    print(err)
                    raise(err)

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
                report.append('Mean ROC AUC and F1: ' + str(f))
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

class GeneticAlgorithm:
    def __init__(self, test_name, param_translator: ProblemTranslator, pop_size, n_jobs=24,
                 metric_name=None, metric_name2=None, crossover_rate=0.8,
                 mutation_rate=0.1, tournament_size=3):
        """
        Genetic Algorithm for optimizing parameters.

        Args:
            test_name (str): Name of the experiment.
            param_translator (ProblemTranslator): Object that translates parameter sets.
            pop_size (int): Size of the population.
            n_jobs (int): Number of parallel processes.
            metric_name (str): Primary metric to optimize.
            metric_name2 (str): Secondary metric to optimize.
            crossover_rate (float): Probability of crossover occurring.
            mutation_rate (float): Probability of mutation occurring.
            tournament_size (int): Number of candidates in tournament selection.
        """
        self.test_name = test_name
        self.param_translator = param_translator
        self.bounds = list(zip(param_translator.lower_bounds, param_translator.upper_bounds))
        self.pop_size = pop_size
        self.n_jobs = n_jobs
        self.metric_name = metric_name
        self.metric_name2 = metric_name2
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        random.seed(1337)
        self._initialize_population()

    def _initialize_population(self):
        """Generates the initial population with random values within bounds."""
        self.population = [[random.uniform(lb, ub) for lb, ub in self.bounds] for _ in range(self.pop_size)]

    def _tournament_selection(self, solutions_with_fitness):
        """Selects individuals using tournament selection."""
        selected = []
        for _ in range(self.pop_size):
            candidates = random.sample(solutions_with_fitness, self.tournament_size)
            candidates.sort(key=lambda x: (round(x[1][0], 2), x[1][1]), reverse=True)  # Maximize fitness
            selected.append(candidates[0])
        return selected

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover between two parents."""
        if random.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()  # No crossover, return as is
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def _mutate(self, individual):
        """Applies mutation by randomly changing some genes within bounds."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                lb, ub = self.bounds[i]
                individual[i] = random.uniform(lb, ub)
        return individual

    def _generate_new_population(self, solutions_with_fitness):
        """Creates the next generation using selection, crossover, and mutation."""
        new_population = []
        selected_parents = self._tournament_selection(solutions_with_fitness)
        
        while len(new_population) < self.pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = self._crossover(parent1[0], parent2[0])
            new_population.append(self._mutate(child1))
            if len(new_population) < self.pop_size:
                new_population.append(self._mutate(child2))
        
        return new_population
    
    def _evaluate_fitness(self, objective_func):
        if self.n_jobs <= 1:
            fitness_vec = [objective_func(a) for a in self.population]
        else:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(self.n_jobs)
            signal.signal(signal.SIGINT, original_sigint_handler)
            try:
                res = pool.map_async(objective_func, self.population)
                fitness_vec = res.get()
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                quit(1)
            else:
                pool.close()
            pool.join()
        return fitness_vec
    
    def run(self, objective_func, generations=4, log_dir="/tmp/"):
        """
        Runs the genetic algorithm for a given number of generations.

        Args:
            objective_func (function): Function to evaluate fitness.
            generations (int): Number of generations.
            log_dir (str): Directory to store log files.

        Returns:
            tuple: Best solution, best fitness, and detailed report.
        """
        all_solutions = []
        report = []
        last_n = 0
        super_log_dir = path.dirname(log_dir)
        print('Base benchmark dir is', super_log_dir)
        super_log_dir = super_log_dir if path.exists(super_log_dir) else None
        gen_bests = []
        for gen in range(generations):
            print(f"{self.test_name} - Generation {gen}")

            # Evaluate fitness
            fitness_results = self._evaluate_fitness(objective_func)

            # Log generation data
            generation_log = {
                'population': [{'solution': self.param_translator.decode(sol), 'metrics': metrics}
                               for sol, metrics in zip(self.population, fitness_results)]
            }
            with open(f"{log_dir}/gen_{gen}_population.json", "w") as f:
                json.dump(generation_log, f, indent=4)
            
            if super_log_dir:
                try:
                    new_n = iterative_gens_draw(super_log_dir, prev_n_gens=last_n)
                    last_n = new_n
                except Exception as err:
                    print('Exception while trying to draw evolution:')
                    print(err)

            # Extract primary and secondary fitness scores
            fitness_values = [(metrics[self.metric_name], metrics[self.metric_name2]) 
                              for metrics in fitness_results]
            main_fitness = sorted([round(a, 3) for a, b in fitness_values])
            current_best = main_fitness[-1]
            gen_bests.append(current_best)
            gens_sorted = sorted([(val, n) for n, val in enumerate(gen_bests)])
            best_gen_so_far = gens_sorted[-1][1]
            
            # Combine population with fitness scores
            solutions_with_fitness = list(zip(self.population, fitness_values))
            all_solutions.extend(solutions_with_fitness)

            # Sort all solutions by fitness (higher is better)
            all_solutions.sort(key=lambda x: (round(x[1][0], 2), x[1][1]), reverse=True)

            # Log top 5 solutions
            report.append(f"\nTop solutions at generation {gen}:")
            report.extend([f"Mean ROC AUC and F1: {fitness}" for _, fitness in all_solutions[:5]])
            if gen - best_gen_so_far >= 2:
                report.append(f"Best gen was {best_gen_so_far}, stopping early")
            # Generate new population (except for last generation)
            if gen < generations - 1:
                self.population = self._generate_new_population(solutions_with_fitness)

        # Final results
        top_solutions = all_solutions[:min(12, len(self.population))]
        best_solution, best_fitness = all_solutions[0]

        report.append(f"Top {len(top_solutions)} final solutions:")
        for sol, fitness in top_solutions:
            solution_str = json.dumps(self.param_translator.decode(sol), indent=4)
            report.extend(solution_str.split("\n"))
            report.append(f"ROC AUC: {fitness}")

        # Log parameter statistics
        for i, param_name in enumerate(self.param_translator.params_list):
            param_values = [sol[i] for sol, _ in top_solutions]
            report.append(f"{param_name} - Mean: {np.mean(param_values):.4f}, Std: {np.std(param_values):.4f}")

        return best_solution, best_fitness, "\n".join(report)