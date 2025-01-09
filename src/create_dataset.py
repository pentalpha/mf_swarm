import sys
from dimension_db import DimensionDB

dataset_types = {
    'base_benchmark',
    'cell_location',
    'full_swarm',
    'small_swarm'
}

if __name__ == "__main__":
    dimension_db_releases_dir = sys.argv[1]
    dimension_db_release_n = sys.argv[2]
    min_proteins_per_mf    = int(sys.argv[3])
    dataset_type           = sys.argv[4]

    assert dataset_type in dataset_types

    dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n)