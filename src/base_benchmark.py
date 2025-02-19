from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import sys
print('New thread', file=sys.stderr)

from tqdm import tqdm
import polars as pl

from metaheuristics import param_bounds
from create_dataset import Dataset, find_latest_dataset
from dimension_db import DimensionDB
from node_factory import MetaheuristicTest, sample_train_test, train_node
from util_base import run_command, plm_sizes

def run_test(exp):
    print('Preparing', exp['name'], exp['features'])
    name = exp['name']
    features = exp['features']
    nodes = exp['nodes']
    experiment_dir = exp['experiment_dir']
    test_perc = exp['test_perc']

    local_dir = experiment_dir + '/' + name + '_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())

    run_command(['mkdir -p', local_dir])

    print('Separating train and test', exp['name'])
    params_list = []
    for node_name, node in tqdm(nodes.items(), total=len(nodes.keys())):
        sample_train_test(node['traintest_path'], features, test_perc)
        params_list.append({
                'test_perc': test_perc,
                'node': node,
                'node_name': node_name
            }
        )

    print('Preparing', exp['name'])
    meta_test = MetaheuristicTest(name, params_list, features, 11)

    print('Running', exp['name'])
    solution_dict, fitness, report = meta_test.test()
    print('Saving', exp['name'])
    meta_report_path = local_dir + '/optimization.txt'
    open(meta_report_path, 'w').write(report)
    json.dump(solution_dict, open(local_dir + '/solution.json', 'w'), indent=4)

    

    return solution_dict

if __name__ == '__main__':
    dimension_db_releases_dir = sys.argv[1]
    #hint: 1
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #hint: 30
    min_proteins_per_mf    = int(sys.argv[4])
    val_perc    = float(sys.argv[5])
    test_perc    = float(sys.argv[6])
    base_benchmark_dir    = sys.argv[7]

    run_command(['mkdir -p', base_benchmark_dir])

    dataset_type           = 'base_benchmark'
    matching_dataset_path = find_latest_dataset(datasets_dir, dataset_type, 
                                            min_proteins_per_mf, dimension_db_release_n,
                                            val_perc)
    if matching_dataset_path != None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=True)
        dataset = Dataset(dimension_db=dimension_db, 
                      min_proteins_per_mf=min_proteins_per_mf, 
                      dataset_type=dataset_type,
                      val_perc=val_perc)
    print('Nodes in dataset:', dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)
    load_order = dataset.datasets_to_load
    load_order.sort(key=lambda n: plm_sizes[n])
    experiments = [
        {
            'name': dataset_type+'_'+name,
            'features': [name],
            'nodes': dataset.go_clusters,
            'test_perc': test_perc,
            'experiment_dir': path.dirname(base_benchmark_dir)
        }
        for name in load_order
    ]

    results = [run_test(exp) for exp in experiments]
    for name, r in zip(load_order, results):
        r['name'] = name
    json.dump(results, open(base_benchmark_dir + '/results1.json', 'w'), indent=4)