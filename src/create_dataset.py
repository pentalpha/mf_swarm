from collections import Counter
from datetime import datetime
from glob import glob
import json
from math import floor
from os import mkdir, path
import random
import sys
import gzip
from tqdm import tqdm
import obonet
import networkx as nx
import numpy as np
import polars as pl

from dimension_db import DimensionDB
from parquet_loading import VectorLoader
from util_base import concat_lists, create_go_labels, run_command

dataset_types = {
    'base_benchmark',
    'taxon_benchmark',
    'cell_location',
    'full_swarm',
    'small_swarm'
}

def find_latest_dataset(datasets_dir, dataset_type, min_proteins_per_mf, release_n, val_perc):
    dataset_paths = glob(datasets_dir+'/'+dataset_type+'_*')
    #TODO sort by modification time
    dataset_paths.sort()

    matching = []
    print('Looking for datasets at', datasets_dir)
    print(dataset_type, min_proteins_per_mf, release_n)
    for p in dataset_paths:
        print('\t'+p)
        dataset_params = json.load(open(path.join(p, 'params.json'), 'r'))
        if all([k in dataset_params for k in ['release_n', 'min_ann', 'val_perc']]):
            if (dataset_params['release_n'] == release_n 
                and int(dataset_params['min_ann']) == min_proteins_per_mf
                and dataset_params['val_perc'] == val_perc):
                matching.append(p)
            else:
                print('\tvalues not matching')
        else:
            print('\tmandatory keys not found')
    
    print('Found:', matching)
    if len(matching) > 0:
        return matching[-1]
    else:
        return None
    
class Dataset:
    def __init__(self, dataset_path = None, dimension_db: DimensionDB = None, min_proteins_per_mf: int = None, 
            dataset_type: str = None, val_perc: float = None) -> None:
        if dataset_path == None:
            self.new_dataset = True
            self.create_new_dataset(dimension_db, min_proteins_per_mf, dataset_type, val_perc)
        else:
            self.start_from_dir(dataset_path)
            self.new_dataset = False
    
    def start_from_dir(self, dataset_path: str):
        self.dataset_params = json.load(open(path.join(dataset_path, 'params.json'), 'r'))
        self.release_n = self.dataset_params['release_n']
        self.min_proteins_per_mf = int(self.dataset_params['min_ann'])
        self.dataset_type = self.dataset_params['dataset_type']
        self.dataset_name = self.dataset_params['dataset_name']
        self.datasets_to_load = self.dataset_params['datasets_to_load']
        self.go_clusters = json.load(gzip.open(path.join(dataset_path, 'go_clusters.json.gz'), 'rt'))
        self.ids = open(path.join(dataset_path, 'ids.txt'), 'r').read().split('\n')
        self.go_ids = open(path.join(dataset_path, 'go_ids.txt'), 'r').read().split('\n')
        self.val_perc = self.dataset_params['val_perc']
    
    def goids_to_cluster(self, cluster_name, cluster_gos, filtered_ann, 
                         parquet_loader, traintest_set, val_set, tmp_dir,
                         dimension_db,
                         max_proteins_traintest=None):
        print('Filtering annotation')
        cluster_ann = {k: v.intersection(cluster_gos) for k, v in filtered_ann.items()}
        cluster_ann = {k: v for k, v in cluster_ann.items() if len(v) > 0}
        
        cluster_ids = [p for p in dimension_db.ids if p in cluster_ann]
        cluster_gos_set = set(cluster_gos)
        print('Creating DF', cluster_name, 'with', len(cluster_ids), 'proteins')
        traintest_ids = set(cluster_ids).intersection(traintest_set)
        val_ids = set(cluster_ids).intersection(val_set)

        if max_proteins_traintest:
            if len(traintest_ids) > max_proteins_traintest:
                parts_len = int(len(cluster_gos)/3)
                first_gos = cluster_gos[:parts_len]
                middle_gos = cluster_gos[parts_len:parts_len+parts_len]
                print('Cluster has ', len(cluster_gos), 
                      ' and priority portion has', len(first_gos))
                #other_gos = cluster_gos[parts_len:]

                print('Sampling proteins for least frequent classes:', first_gos)
                less_freq_proteins = set()
                for prot_id in list(traintest_ids):
                    if len(cluster_ann[prot_id].intersection(first_gos)) > 0:
                        traintest_ids.remove(prot_id)
                        less_freq_proteins.add(prot_id)
                
                remaining_n = max_proteins_traintest - len(less_freq_proteins)
                print(len(less_freq_proteins), 'less_freq_proteins')
                print('collecting more', remaining_n, 'proteins')

                print('Sampling proteins for middle frequent classes:', middle_gos)
                middle_freq_proteins = set()
                for prot_id in list(traintest_ids):
                    if len(cluster_ann[prot_id].intersection(middle_gos)) > 0:
                        #traintest_ids.remove(prot_id)
                        middle_freq_proteins.add(prot_id)
                
                for_middle = int(remaining_n/2)
                print(len(middle_freq_proteins), 'middle_freq_proteins')
                print(for_middle, 'for_middle')
                if for_middle < len(middle_freq_proteins):
                    middle_ids = random.sample(list(middle_freq_proteins), for_middle)
                else:
                    middle_ids = middle_freq_proteins
                traintest_ids = traintest_ids.difference(middle_ids)
                
                print('combining lists')
                remaining_ids = random.sample(list(traintest_ids), 
                    max_proteins_traintest - len(less_freq_proteins) - len(middle_ids))
                traintest_ids = list(less_freq_proteins) + list(middle_ids) + remaining_ids
                random.shuffle(traintest_ids)

                assert len(traintest_ids) == max_proteins_traintest, [len(l) for l in [less_freq_proteins, middle_ids, remaining_ids]]
        #quit(1)
        print('Counting proteins at each class')
        ann_by_go = {goid: {'traintest': [], 'val': []} for goid in cluster_gos_set}
        for l, key in [(traintest_ids, 'traintest'), (val_ids, 'val')]:
            for protein_id in tqdm(l):
                prot_ann = cluster_ann[protein_id]
                for goid in prot_ann.intersection(cluster_gos_set):
                    ann_by_go[goid][key].append(protein_id)
        
        to_remove = []
        for goid in cluster_gos:
            go_traintests = ann_by_go[goid]['traintest']
            go_val = ann_by_go[goid]['val']
            print(goid, len(go_traintests), len(go_val))
            if len(go_traintests) < 3:
                print(goid, 'has zero samples in traintest set')
                to_remove.append(goid)
            elif len(go_val) < 3:
                print(goid, 'has zero samples in validation set')
                to_remove.append(goid)
        
        print('Loading parquet')

        cluster_df = parquet_loader.load_vectors_by_ids(
            list(traintest_ids) + list(val_ids), 
            self.datasets_to_load,
            remove_na=True)
        loaded_ids = cluster_df['id'].to_list()
        print('Creating labels for dataset df')
        labels, cluster_go_proper_order = create_go_labels(loaded_ids, cluster_ann)
        cluster_df = cluster_df.with_columns(
            labels=labels
        )
        print(len(cluster_df), 'proteins in cluster df')
        traintest_df = cluster_df.filter(pl.col('id').is_in(set(traintest_ids)))
        val_df = cluster_df.filter(pl.col('id').is_in(val_set))
        print(len(traintest_df), 'proteins in cluster traintest df')
        print(len(val_df), 'proteins in cluster val df')
        #cluster_df['labels'] = labels_np

        print('Saving parquet to tmp')
        traintest_df_path = tmp_dir + '/traintest_' + self.dataset_name + '-' + cluster_name + '.parquet'
        val_df_path = tmp_dir + '/val_' + self.dataset_name + '-' + cluster_name + '.parquet'
        traintest_df.write_parquet(traintest_df_path)
        val_df.write_parquet(val_df_path)
        del traintest_df
        del val_df
        del cluster_df

        cluster = {
            'id': loaded_ids,
            'go': cluster_go_proper_order,
            'traintest_path': traintest_df_path,
            'val_path': val_df_path
        }
        
        return cluster

    def create_new_dataset(self, dimension_db: DimensionDB, min_proteins_per_mf: int, 
            dataset_type: str, val_perc: float):
        tmp_dir = path.expanduser('~/tmp')
        if not path.exists(tmp_dir):
            mkdir(tmp_dir)
        
        self.release_n = dimension_db.release_dir.split("_")[-1]
        self.min_proteins_per_mf = min_proteins_per_mf
        self.dataset_type = dataset_type
        self.dataset_name = dataset_type + '_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
        self.val_perc = val_perc

        self.dataset_params = {'release_n': self.release_n, 
            'min_ann': str(min_proteins_per_mf), 
            'dataset_type': dataset_type,
            'dataset_name': self.dataset_name,
            'val_perc': val_perc}
        traintest_set, val_set, filtered_ann, go_freqs = dimension_db.get_proteins_set(min_proteins_per_mf, val_perc)

        max_proteins_traintest = None
        if dataset_type == 'base_benchmark' or 'taxon_benchmark':
            go_clusters = Dataset.base_benchmark_goids_clustering(dimension_db, go_freqs)
            max_proteins_traintest = 4500
            if dataset_type == 'base_benchmark':
                self.datasets_to_load = dimension_db.plm_names
            else:
                self.datasets_to_load = ['ankh_base', 'prottrans'] + dimension_db.taxa_onehot_names + dimension_db.taxa_profile_names
        elif dataset_type == "full_swarm":
            go_clusters = Dataset.full_mf_goids_clustering(dimension_db, go_freqs, len(traintest_set))
            self.datasets_to_load = ['taxa_256', 'ankh_base', 'prottrans']
        self.dataset_params['datasets_to_load'] = self.datasets_to_load

        self.go_clusters = {}
        parquet_loader = VectorLoader(dimension_db.release_dir)
        print('Making datasets for clusters')

        for cluster_name, cluster_gos in tqdm(go_clusters.items(), total = len(go_clusters)):
            self.go_clusters[cluster_name] = self.goids_to_cluster(cluster_name, cluster_gos, filtered_ann, 
                parquet_loader, traintest_set, val_set, tmp_dir, dimension_db, 
                max_proteins_traintest=max_proteins_traintest)

        for clustername in self.go_clusters.keys():
            print(clustername, 'cluster created')
        
        self.go_ids = sorted(set(concat_lists([g['go'] for g in self.go_clusters.values()])))
        all_ids = set(concat_lists([g['id'] for g in self.go_clusters.values()]))
        self.ids = [p for p in dimension_db.ids if p in all_ids]
        
        print(len(self.ids), 'proteins')
        print(len(self.go_ids), 'go ids')

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
        to_use = test_nodes[level] if level in test_nodes else []
        
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
                if current_percentile in to_use or not only_test_nodes:
                    to_keep.append(cluster_name)
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
                if current_percentile in to_use or not only_test_nodes:
                    to_keep.append(cluster_name)
                level_clusters[cluster_name] = new_cluster
                del level_clusters[last_cluster_name]
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
        test_nodes = {4: [0], 5: [0], 6: [0, 1], 7: [0, 1, 2]}
        for level_name, goids in levels.items():
            new_clusters, new_to_keep = Dataset.full_mf_goids_clustering_level_iteration(level_name, goids, go_freqs, 
                test_nodes, n_proteins, percentiles, only_test_nodes=is_test)
            for n in new_to_keep:
                clusters[n] = new_clusters[n]
        
        return clusters
    
    def save(self, datasets_dir):
        outputdir = path.join(datasets_dir, self.dataset_name)
        if not path.exists(outputdir):
            run_command(['mkdir -p', outputdir])
        json.dump(self.dataset_params, open(path.join(outputdir, 'params.json'), 'w'), indent=4)
        for cluster_name, cluster_dict in self.go_clusters.items():
            df_path1_tmp = cluster_dict['traintest_path']
            df_path1 = outputdir + '/traintest_' + cluster_name + '.parquet'
            run_command(['mv', df_path1_tmp, df_path1])
            cluster_dict['traintest_path'] = df_path1

            df_path2_tmp = cluster_dict['val_path']
            df_path2 = outputdir + '/val_' + cluster_name + '.parquet'
            run_command(['mv', df_path2_tmp, df_path2])
            cluster_dict['val_path'] = df_path2

        json.dump(self.go_clusters, 
            gzip.open(path.join(outputdir, 'go_clusters.json.gz'), 'wt'), indent=4)
        ids_str = '\n'.join(self.ids)
        open(path.join(outputdir, 'ids.txt'), 'w').write(ids_str)
        gos_str = '\n'.join(self.go_ids)
        open(path.join(outputdir, 'go_ids.txt'), 'w').write(gos_str)

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