import sys
from mf_swarm_lib.utils.swarm_parsing import read_cluster_of_goids
from mf_swarm_lib.utils.plotting import plot_hierarchy

if __name__ == "__main__":
    swarm_dir = sys.argv[1]
    pddb_dir = sys.argv[2]
    
    print('Reading swarm')
    cluster_n, distances_original, protein_ids = read_cluster_of_goids(swarm_dir)
    
    print('Plotting')
    plot_hierarchy(swarm_dir, pddb_dir, cluster_n, protein_ids)
