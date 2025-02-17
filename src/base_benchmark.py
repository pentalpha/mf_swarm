import sys

from create_dataset import Dataset, find_latest_dataset
from dimension_db import DimensionDB

if __name__ == '__main__':
    dimension_db_releases_dir = sys.argv[1]
    #hint: 1
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #hint: 30
    min_proteins_per_mf    = int(sys.argv[4])

    dataset_type           = 'base_benchmark'
    matching_dataset_path = find_latest_dataset(datasets_dir, dataset_type, 
                                            min_proteins_per_mf, dimension_db_release_n)
    if matching_dataset_path != None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=False)
        dataset = Dataset(dimension_db=dimension_db, 
                      min_proteins_per_mf=min_proteins_per_mf, 
                      dataset_type=dataset_type)
    print('Nodes in dataset:', dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)