from collections import Counter
from datetime import datetime
import json
from math import floor
from os import mkdir, path
import sys
import gzip
from tqdm import tqdm
import obonet
import networkx as nx
import numpy as np

from dimension_db import DimensionDB
from parquet_loading import VectorLoader
from util_base import concat_lists, create_go_labels, run_command

dataset_types = {
    'base_benchmark',
    'cell_location',
    'full_swarm',
    'small_swarm'
}

#molecular function root
irrelevant_mfs = {'GO:0003674'}

def load_annotation_terms(file_path: str):
    by_protein = {}
    for rawline in gzip.open(file_path, 'rt'):
        cells = rawline.rstrip('\n').split("\t")
        uniprot = cells[0]
        mfs = cells[1].split(',')
        mfs = [x for x in mfs if x.startswith("GO:")]
        by_protein[uniprot] = mfs

    return by_protein

class Dataset:
    def __init__(self, dataset_path = None, dimension_db: DimensionDB = None, 
                 min_proteins_per_mf: int = None, dataset_type: str = None) -> None:
        if dataset_path == None:
            self.new_dataset = True
            self.create_new_dataset(dimension_db, min_proteins_per_mf, dataset_type)
        else:
            self.start_from_dir(dataset_path)
            self.new_dataset = False
    
    def start_from_dir(self, dataset_path: str):
        self.dataset_params = json.load(open(path.join(dataset_path, 'params.json'), 'r'))
        self.release_n = self.dataset_params['release_n']
        self.min_proteins_per_mf = int(self.dataset_params['min_proteins_per_mf'])
        self.dataset_type = self.dataset_params['dataset_type']
        self.dataset_name = self.dataset_params['dataset_name']
        self.datasets_to_load = self.dataset_params['datasets_to_load']
        self.go_clusters = json.load(gzip.open(path.join(dataset_path, 'go_clusters.json.gz'), 'rt'))
        self.ids = open(path.join(dataset_path, 'ids.txt'), 'r').read().split('\n')
        self.go_ids = open(path.join(dataset_path, 'go_ids.txt'), 'r').read().split('\n')
    
    def create_new_dataset(self, dimension_db: DimensionDB, 
                           min_proteins_per_mf: int, dataset_type: str):
        tmp_dir = path.expanduser('~/tmp')
        if not path.exists(tmp_dir):
            mkdir(tmp_dir)
        
        self.release_n = dimension_db.release_dir.split("_")[-1]
        self.min_proteins_per_mf = min_proteins_per_mf
        self.dataset_type = dataset_type
        self.dataset_name = dataset_type + '_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
        
        self.dataset_params = {'release_n': self.release_n, 
            'min_ann': str(min_proteins_per_mf), 
            'dataset_type': dataset_type,
            'dataset_name': self.dataset_name}

        filtered_ann, go_freqs = self.create_filtered_annotation(dimension_db, min_proteins_per_mf)

        parquet_loader = VectorLoader(dimension_db.release_dir)
        if dataset_type == 'base_benchmark':
            go_clusters = Dataset.base_benchmark_goids_clustering(dimension_db, go_freqs)
            self.datasets_to_load = dimension_db.plm_names
        self.dataset_params['datasets_to_load'] = self.datasets_to_load
        self.go_clusters = {}
        print('Making datasets for clusters')
        for cluster_name, cluster_gos in tqdm(go_clusters.items(), total = len(go_clusters)):
            print('Filtering annotation')
            cluster_ann = {k: v.intersection(cluster_gos) for k, v in filtered_ann.items()}
            no_ann_ids = [k for k, v in cluster_ann.items() if len(v) == 0]
            for protein_id in no_ann_ids:
                del cluster_ann[protein_id]
            
            cluster_ids = [p for p in dimension_db.ids if p in cluster_ann]
            print('Creating DF')
            cluster_df = parquet_loader.load_vectors_by_ids(cluster_ids, self.datasets_to_load,
                                                        remove_na=True)
            
            print('Deleting IDs who could not be loaded')
            loaded_ids = cluster_df['id'].to_list()
            for x in cluster_ids:
                if not x in loaded_ids:
                    del cluster_ann[x]
            
            print('Creating labels for dataset df')
            labels, cluster_go_proper_order = create_go_labels(loaded_ids, cluster_ann)
            cluster_df = cluster_df.with_columns(
                labels=labels
            )
            #cluster_df['labels'] = labels_np

            print('Saving parquet to tmp')
            df_path = tmp_dir + '/' + self.dataset_name + '-' + cluster_name + '.parquet'
            print(len(loaded_ids), 'proteins')
            print(cluster_df)
            cluster_df.write_parquet(df_path)
            del cluster_df

            self.go_clusters[cluster_name] = {
                'id': loaded_ids,
                'go': cluster_go_proper_order,
                'df_path': df_path
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
        for l in [4, 5, 6, 7]:
            gos = levels[l]
            gos.sort(key=lambda g: go_freqs[g])
            print(len(gos), 'GO IDs at level', l)
            '''if not l in [1, '1']:
                #print('not level 1')
                index = int(len(gos)*(top_worst_perc/100))
                worst_gos = gos[:index]
            else:
                worst_gos = gos[:8]'''
            worst_gos = gos[:60]
            last_min_freq = go_freqs[worst_gos[0]]
            max_freq = go_freqs[worst_gos[-1]]
            cluster_name = ('Level-'+str(l)+'_Freq-'+str(last_min_freq)+'-'
                    +str(max_freq)+'_N-'+str(len(worst_gos)))
            clusters[cluster_name] = worst_gos

        return clusters


    def create_filtered_annotation(self, dimension_db: DimensionDB, min_proteins_per_mf: int):
        all_swissprot_ids = set(dimension_db.ids)

        all_annotations = load_annotation_terms(file_path=dimension_db.mf_gos_path)
        all_proteins = set(all_annotations.keys())
        print(len(all_proteins), 'proteins in', dimension_db.mf_gos_path)
        all_goids = set(concat_lists([list(v) for v in all_annotations.values()]))
        print(len(all_goids), 'GO IDs in', dimension_db.mf_gos_path)

        all_annotations = {k: set(v) for k, v in all_annotations.items() if k in all_swissprot_ids}
        all_proteins = set(all_annotations.keys())
        print(len(all_proteins), 'uniprot proteins in', dimension_db.mf_gos_path)
        goid_list = concat_lists([list(v) for v in all_annotations.values()])
        goid_list = [goid for goid in goid_list if not goid in irrelevant_mfs]
        print(len(set(goid_list)), 'GO IDs from uniprot proteins in', dimension_db.mf_gos_path)

        go_counts = Counter(goid_list)
        frequent_go_ids = set()
        for goid, freq in go_counts.items():
            if freq >= min_proteins_per_mf:
                frequent_go_ids.add(goid)
        go_counts_list = [(n, go) for go, n in go_counts.items()]
        go_counts_list.sort(reverse=True)
        tops = 10
        while tops >= 0:
            print(tops, 'top term:', go_counts_list[tops])
            tops -= 1

        print(len(frequent_go_ids), 'frequent GO IDs from uniprot proteins in', dimension_db.mf_gos_path)

        for k in all_annotations.keys():
            all_annotations[k] = all_annotations[k].intersection(frequent_go_ids)
        
        all_annotations = {k: v for k, v in all_annotations.items() if len(v) > 0}
        frequent_prots = set(all_annotations.keys())
        print(len(frequent_prots), 'uniprot proteins with frequent GOs in', dimension_db.mf_gos_path)
        go_freqs_filtered = {g: c for g, c in go_counts.items() if g in frequent_go_ids}
        return all_annotations, go_freqs_filtered
    
    def save(self, datasets_dir):
        outputdir = path.join(datasets_dir, self.dataset_name)
        if not path.exists(outputdir):
            run_command(['mkdir -p', outputdir])
        json.dump(self.dataset_params, open(path.join(outputdir, 'params.json'), 'w'), indent=4)
        for cluster_name, cluster_dict in self.go_clusters.items():
            df_path_tmp = cluster_dict['df_path']
            df_path = outputdir + '/' + cluster_name + '.parquet'
            run_command(['mv', df_path_tmp, df_path])
            cluster_dict['df_path'] = df_path

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