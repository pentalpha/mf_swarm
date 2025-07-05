import sys
import json

if __name__ == "__main__":
    benchmark_file = sys.argv[1]
    param_bounds_file = sys.argv[2]
    #benchmark_file = '/home/pita/data/mf_swarm_datasets/benchmark.tsv'
    benchmark_lines = open(benchmark_file, 'r').read().split('\n')

    param_solutions = []
    for rawline in benchmark_lines:
        columns = rawline.split('\t')
        if len(columns) == 22:
            try:
                sol = json.loads(columns[-1].strip('"').replace('""', '"'))
                auprc_w = float(columns[4])
                param_solutions.append((sol, auprc_w))
            except Exception  as err:
                print(err)
                print(columns)
    
    metaparam_values = {}
    to_keep = ['final', 'esm2_t33', 'ankh_base', 'prottrans', 'taxa_256']
    for params, auprc_w in param_solutions:
        for model_part, sub_params in params.items():
            if model_part in to_keep:
                for param_name, param_value in sub_params.items():
                    full_name = model_part + ';;' + param_name
                    if full_name not in metaparam_values:
                        metaparam_values[full_name] = set()
                    metaparam_values[full_name].add((param_value, auprc_w))
    
    for param_name, values in metaparam_values.items():
        print(param_name, values)

    param_bounds = {}
    for param_name, values in metaparam_values.items():
        values_sorted = sorted(values, key = lambda x: x[1])
        best_value = values_sorted[-1][0]
        values_list = [x for x, y in values]
        lesser = min(values_list)
        greater = max(values_list)
        model_part, param_name = param_name.split(';;')

        if model_part not in param_bounds:
            param_bounds[model_part] = {}
        param_range = greater - lesser
        if param_range > 0:
            to_increase = param_range*0.15
            new_lesser = best_value - to_increase
            new_greater = best_value + to_increase
            param_bounds[model_part][param_name] = [new_lesser, new_greater]
        else:
            span = best_value * 0.15
            new_lesser = best_value - span
            new_greater = best_value + span
            param_bounds[model_part][param_name] = [new_lesser, new_greater]
    
    print(json.dumps(param_bounds, indent=4))

    json.dump(param_bounds, open(param_bounds_file, 'w'), indent=4)
