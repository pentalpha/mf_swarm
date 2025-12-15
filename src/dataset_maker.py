
import sys
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.data.dataset import Dataset

if __name__ == "__main__":
    dimension_db_releases_dir = sys.argv[1]
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #30
    min_proteins_per_mf    = int(sys.argv[4])
    dataset_type           = sys.argv[5]

    assert dataset_type in dataset_types

    dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=False)
    dataset = Dataset(dimension_db=dimension_db, min_proteins_per_mf=min_proteins_per_mf, dataset_type=dataset_type)
    if dataset.new_dataset:
        dataset.save(datasets_dir)