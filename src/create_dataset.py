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
from util_base import concat_lists, run_command

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
            self.create_new_dataset(dimension_db, min_proteins_per_mf,  )
        else:
            self.start_from_dir(dataset_path)
            self.new_dataset = False
    
    def start_from_dir(dataset_path: str):
        pass

    def create_new_dataset(self, dimension_db: DimensionDB, 
                           min_proteins_per_mf: int, dataset_type: str):
        self.dataset_params = {'release_n': dimension_db.release_dir.split("_")[-1], 'min_ann': str(min_proteins_per_mf), 'dataset_type': dataset_type}
        self.dataset_name = dataset_type + '_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
        filtered_ann, go_freqs = self.create_filtered_annotation(dimension_db, min_proteins_per_mf)

        parquet_loader = VectorLoader(dimension_db.release_dir)
        if dataset_type == 'base_benchmark':
            self.go_clusters = Dataset.base_benchmark_goids_clustering(dimension_db, go_freqs)
            self.datasets_to_load = parquet_loader.plm_names
        
        for clustername in self.go_clusters.keys():
            print(clustername, 'cluster created')
        
        self.go_ids = concat_lists(self.go_clusters.values())
        for k in filtered_ann.keys():
            filtered_ann[k] = filtered_ann[k].intersection(self.go_ids)
        filtered_ann = {k: v for k, v in filtered_ann.items() if len(v) > 0}
        frequent_prots = set(filtered_ann.keys())

        self.ids = [p for p in dimension_db.ids if p in filtered_ann]
        self.ann_dict = {p: sorted(filtered_ann[p], key=lambda g: go_freqs[g]) for p in self.ids}
        print(len(self.ids), 'proteins')
        print(len(self.go_ids), 'go ids')

        #find go IDs for each cluster
        #create parquet file for each cluster
        #filter parquet by removing lines with NaN
        #update ids list by removing ids not found anymore

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
        json.dump(self.ann_dict, gzip.open(path.join(outputdir, 'mf_ann.json.gz'), 'wt'), indent=4)
        ids_str = '\n'.join(self.ids)
        open(path.join(outputdir, 'ids.txt'), 'w').write(ids_str)

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