import sys
import os
from os import path

from mf_swarm_lib.training.cv_testing import test_nodes_from_full_clustering
from mf_swarm_lib.utils.util_base import run_command

if __name__ == "__main__":
    dimension_db_release_n = "1"
    dimension_db_releases_dir = path.expanduser("~/data/protein_dimension_db/")
    datasets_dir = path.expanduser("~/data/mf_swarm_datasets/")
    benchmark_dir = "tmp"
    local_dir = benchmark_dir + "/cv_validation"

    run_command(["mkdir -p", local_dir])

    min_proteins_per_mf = 40
    val_perc = 0.15
    #test_benchmarking_node(datasets_dir, min_proteins_per_mf, dimension_db_release_n,
    #    val_perc, dimension_db_releases_dir, )

    dataset, node, annot_model, params_dict = test_nodes_from_full_clustering(
        datasets_dir, min_proteins_per_mf, 
        dimension_db_release_n, val_perc, dimension_db_releases_dir)