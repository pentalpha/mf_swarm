from tqdm import tqdm
from math import floor
import numpy as np
import obonet
import networkx as nx

def base_benchmark_goids_clustering(dimension_db, go_freqs, go_section_len=24):
    go_graph = obonet.read_obo(dimension_db.go_basic_path)
    #root = 'GO:0003674'
    #go_levels_2 = {}
    #go_n_annotations = {}
    all_goids = list(go_freqs.keys())
    valid_goids = [x for x in all_goids if x in go_graph]
    valid_goids.sort(key=lambda goid: go_freqs[goid])

    worst_gos = valid_goids[:go_section_len]
    best = valid_goids[-go_section_len:]
    middle = int(len(valid_goids)/2)
    mid_len = int((go_section_len/2))
    mid_start = middle-mid_len
    middle_gos = valid_goids[mid_start:mid_start+go_section_len]

    cluster_goids = worst_gos + middle_gos + best
    cluster_name = 'WORST_MID_BEST_'+str(go_section_len)+'-N'+str(len(cluster_goids))

    clusters = {cluster_name: cluster_goids}

    '''for goid in tqdm(valid_goids):
        n_annots = go_freqs[goid]
        
        simple_paths = nx.all_simple_paths(go_graph, source=goid, target=root)
        simple_path_lens = [len(p) for p in simple_paths]
        try:
            mean_dist = floor(np.mean(simple_path_lens)-1)
            go_levels_2[goid] = min(7, mean_dist)
            go_n_annotations[goid] = n_annots
        except ValueError as err:
            print(simple_path_lens)
            print('No path from', goid, 'to', root)
            print(err)
            raise(err)
    
    levels = {l: [] for l in set(go_levels_2.values())}
    for goid, level in go_levels_2.items():
        levels[level].append(goid)

    clusters = {}
    
    for l in [5, 6, 7]:
        gos = levels[l]
        gos.sort(key=lambda g: go_freqs[g])
        print(len(gos), 'GO IDs at level', l)
        
        worst_gos = gos[:50]
        last_min_freq = go_freqs[worst_gos[0]]
        max_freq = go_freqs[worst_gos[-1]]
        cluster_name = ('Level-'+str(l)+'_Freq-'+str(last_min_freq)+'-'
                +str(max_freq)+'_N-'+str(len(worst_gos)))
        clusters[cluster_name] = worst_gos'''
    
    return clusters

def full_mf_goids_clustering_level_iteration(level, goids, go_freqs, 
        test_nodes, n_proteins, percentiles, only_test_nodes=False):
    level_go_freqs = [(go, go_freqs[go]) for go in goids 
        if (go_freqs[go] / n_proteins) < 0.9]
    level_go_freqs.sort(key=lambda tp: tp[1])
    print('Level', level, 'GO Frequencies:', level_go_freqs)
    print('Proteins:', n_proteins, 'Percentiles:', percentiles)
    
    #print('Counting percentiles')
    perc_index = []
    last_index = -1
    for perc in percentiles:
        index = int(len(level_go_freqs)*(perc/100))
        perc_index.append((last_index+1, index))
        last_index = index 
    perc_index.append((last_index+1, len(level_go_freqs)-1))   
    #print(perc_index)
    
    total_len = 0
    last_cluster_name = None
    current_level_test_node_names = test_nodes[level] if level in test_nodes else []
    
    current_percentile = 0
    to_keep = []
    level_clusters = {}
    for start, end in perc_index:
        sub_gos = level_go_freqs[start:end+1]
        min_freq = sub_gos[0][1]
        max_freq = sub_gos[-1][1]
        #print(len(sub_gos))
        total_len += len(sub_gos)
        cluster_goids = [x for x, y in sub_gos]
        
        if len(sub_gos) > 2:
            cluster_name = ('Level-'+str(level)+'_Freq-'+str(min_freq)+'-'
                +str(max_freq)+'_N-'+str(len(sub_gos)))
            if only_test_nodes:
                if current_percentile in current_level_test_node_names:
                    to_keep.append(cluster_name)
            else:
                to_keep.append(cluster_name)
            #if current_percentile in current_level_test_node_names or not only_test_nodes:
            #    to_keep.append(cluster_name)
            level_clusters[cluster_name] = cluster_goids
            #print(cluster_name.split('_'))
        else:
            last_cluster = level_clusters[last_cluster_name]
            level_str, freq_str, n_str = last_cluster_name.split('_')
            _, last_min_str, _ = freq_str.split('-')
            last_min_freq = int(last_min_str)
            new_cluster = last_cluster + cluster_goids
            cluster_name = ('Level-'+str(level)+'_Freq-'+str(last_min_freq)+'-'
                +str(max_freq)+'_N-'+str(len(new_cluster)))
            if only_test_nodes:
                if current_percentile in current_level_test_node_names:
                    to_keep.append(cluster_name)
            else:
                to_keep.append(cluster_name)
            level_clusters[cluster_name] = new_cluster
            del level_clusters[last_cluster_name]
            if last_cluster_name in to_keep:
                to_keep.remove(last_cluster_name)
            #print(cluster_name.split('_'))
        last_cluster_name = cluster_name
        current_percentile += 1
    
    return level_clusters, to_keep
    

def full_mf_goids_clustering(dimension_db, go_freqs, n_proteins, 
        percentiles = [40, 70, 90], is_test=False):
    go_graph = obonet.read_obo(dimension_db.go_basic_path)
    all_goids = list(go_freqs.keys())
    valid_goids = [x for x in all_goids if x in go_graph]
    valid_goids.sort(key=lambda goid: go_freqs[goid])

    root = 'GO:0003674'
    go_levels_2 = {}
    print('Finding paths from MF terms to MF Root')
    for goid in tqdm(valid_goids):
        if goid != root:
            simple_paths = nx.all_simple_paths(go_graph, source=goid, target=root)
            simple_path_lens = [len(p) for p in simple_paths]
            try:
                mean_dist = floor(np.mean(simple_path_lens)-1)
                go_levels_2[goid] = min(7, mean_dist)
            except ValueError as err:
                print(simple_path_lens)
                print('No path from', goid, 'to', root)
                print(err)
                raise(err)
    
    levels = {l: [] for l in set(go_levels_2.values())}
    for goid, level in go_levels_2.items():
        levels[level].append(goid)
    del go_levels_2

    clusters = {}
    if is_test:
        test_nodes = {4: [0], 5: [0], 6: [0, 1], 7: [0, 1, 2]}
    else:
        test_nodes = {}
    for level_name, goids in levels.items():
        new_clusters, new_to_keep = full_mf_goids_clustering_level_iteration(
            level_name, goids, go_freqs, 
            test_nodes, n_proteins, percentiles, only_test_nodes=is_test)
        for n in new_to_keep:
            clusters[n] = new_clusters[n]
    
    return clusters
