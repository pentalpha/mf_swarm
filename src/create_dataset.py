from collections import Counter
from datetime import datetime
from glob import glob
import json
from math import floor
from os import mkdir, path
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

        if dataset_type == 'base_benchmark':
            go_clusters = Dataset.base_benchmark_goids_clustering(dimension_db, go_freqs)
            self.datasets_to_load = dimension_db.plm_names
        self.dataset_params['datasets_to_load'] = self.datasets_to_load

        self.go_clusters = {}
        parquet_loader = VectorLoader(dimension_db.release_dir)
        print('Making datasets for clusters')
        for cluster_name, cluster_gos in tqdm(go_clusters.items(), total = len(go_clusters)):
            print('Filtering annotation')
            cluster_ann = {k: v.intersection(cluster_gos) for k, v in filtered_ann.items()}
            cluster_ann = {k: v for k, v in cluster_ann.items() if len(v) > 0}
            
            cluster_ids = [p for p in dimension_db.ids if p in cluster_ann]
            print('Creating DF', cluster_name, 'with', len(cluster_ids), 'proteins')
            cluster_df = parquet_loader.load_vectors_by_ids(cluster_ids, self.datasets_to_load,
                                                        remove_na=True)
            
            print('Deleting IDs who could not be loaded')
            loaded_ids = cluster_df['id'].to_list()
            loaded_ids_set = set(loaded_ids)
            could_not_be_loaded = 0
            for x in cluster_ids:
                if not x in loaded_ids_set:
                    del cluster_ann[x]
                    could_not_be_loaded += 1
            assert could_not_be_loaded == 0
            
            print('Creating labels for dataset df')
            labels, cluster_go_proper_order = create_go_labels(loaded_ids, cluster_ann)
            cluster_df = cluster_df.with_columns(
                labels=labels
            )
            print(len(cluster_df), 'proteins in cluster df')
            traintest_df = cluster_df.filter(pl.col('id').is_in(traintest_set))
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

            self.go_clusters[cluster_name] = {
                'id': loaded_ids,
                'go': cluster_go_proper_order,
                'traintest_path': traintest_df_path,
                'val_path': val_df_path
            }

        for clustername in self.go_clusters.keys():
            print(clustername, 'cluster created')
        
        self.go_ids = sorted(set(concat_lists([g['go'] for g in self.go_clusters.values()])))
        all_ids = set(concat_lists([g['id'] for g in self.go_clusters.values()]))
        self.ids = [p for p in dimension_db.ids if p in all_ids]
        
        print(len(self.ids), 'proteins')
        print(len(self.go_ids), 'go ids')

    def base_benchmark_goids_clustering(dimension_db, go_freqs, top_worst_perc=35):
        go_graph = obonet.read_obo(dimension_db.go_basic_path)
        root = 'GO:0003674'
        go_levels_2 = {}
        go_n_annotations = {}
        all_goids = list(go_freqs.keys())
        valid_goids = [x for x in all_goids if x in go_graph]
        for goid in tqdm(valid_goids):
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

        '''for l in [3, 4]:
            gos = levels[l]
            gos.sort(key=lambda g: go_freqs[g], reverse=True)
            print(len(gos), 'GO IDs at level', l)
            best_gos = gos[:8]
            last_min_freq = go_freqs[best_gos[-1]]
            max_freq = go_freqs[best_gos[0]]
            cluster_name = ('Level-'+str(l)+'_Freq-'+str(last_min_freq)+'-'
                    +str(max_freq)+'_N-'+str(len(best_gos)))
            clusters[cluster_name] = best_gos'''
        
        for l in [5, 6, 7]:
            gos = levels[l]
            gos.sort(key=lambda g: go_freqs[g])
            print(len(gos), 'GO IDs at level', l)
            '''if not l in [1, '1']:
                #print('not level 1')
                index = int(len(gos)*(top_worst_perc/100))
                worst_gos = gos[:index]
            else:
                worst_gos = gos[:8]'''
            worst_gos = gos[:50]
            last_min_freq = go_freqs[worst_gos[0]]
            max_freq = go_freqs[worst_gos[-1]]
            cluster_name = ('Level-'+str(l)+'_Freq-'+str(last_min_freq)+'-'
                    +str(max_freq)+'_N-'+str(len(worst_gos)))
            clusters[cluster_name] = worst_gos
        
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