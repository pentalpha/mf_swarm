from collections import Counter
from pickle import dump
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
from sklearn import metrics

from mf_swarm_lib.core.ml.custom_statistics import create_random_baseline
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.data.parquet_loading import VectorLoader
from mf_swarm_lib.utils.util_base import concat_lists, create_go_labels, run_command
from mf_swarm_lib.data.folding import validate_folds, prepare_global_feature_folds, prepare_label_folds_obj
from mf_swarm_lib.data.clustering import base_benchmark_goids_clustering, full_mf_goids_clustering
from mf_swarm_lib.data import validation

dataset_types = {
    'base_benchmark',
    'taxon_benchmark',
    'taxon_benchmark_cv',
    'cell_location',
    'full_swarm',
    'small_swarm'
}

test_types = {
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
                and dataset_params['val_perc'] == val_perc
                and dataset_params['dataset_type'] == dataset_type):
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

    # validate_folds moved to folding.py


# prepare_global_feature_folds moved to folding.py


class Dataset:
    def __init__(self, dataset_path = None, dimension_db: DimensionDB = None, 
            min_proteins_per_mf: int = None, 
            dataset_type: str = None, val_perc: float = None, n_folds = 5) -> None:
        self.n_folds = n_folds
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
    
    def sample_proteins(self, max_proteins_traintest, traintest_ids, cluster_gos,
            cluster_ann):
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

        return traintest_ids

    def goids_to_cluster(self, cluster_name, cluster_gos, filtered_ann, 
                         traintest_ids, traintest_ids_with_folds, val_ids, 
                         tmp_dir, dimension_db,
                         folds_description: dict,
                         max_proteins_traintest=None,
                         n_folds=5):
        print('Filtering annotation')
        cluster_ann = {k: v.intersection(cluster_gos) for k, v in filtered_ann.items()}
        cluster_ann = {k: v for k, v in cluster_ann.items() if len(v) > 0}
        
        
        # Use all available IDs (both positive and negative samples)
        #traintest_ids = list(traintest_ids_with_folds.keys())
        # val_ids is already a list passed in
        
        cluster_gos_set = set(cluster_gos)
        print('Creating DF', cluster_name, 'with', len(traintest_ids) + len(val_ids), 'proteins')

        
        print('Counting proteins at each class')
        ann_by_go = {goid: {'traintest': [], 'val': []} for goid in cluster_gos_set}
        for l, key in [(traintest_ids, 'traintest'), (val_ids, 'val')]:
            for protein_id in tqdm(l):
                if protein_id in cluster_ann:
                    prot_ann = cluster_ann[protein_id]
                    for goid in prot_ann.intersection(cluster_gos_set):
                        ann_by_go[goid][key].append(protein_id)
        
        to_remove = []
        for goid in cluster_gos:
            go_traintests = ann_by_go[goid]['traintest']
            go_val = ann_by_go[goid]['val']
            # print(goid, len(go_traintests), len(go_val))
            if len(go_traintests) < 4:
                print(goid, 'has few samples in traintest set')
                to_remove.append(goid)
            elif len(go_val) < 4:
                print(goid, 'has few samples in validation set')
                to_remove.append(goid)
        
        print('Creating labels for dataset df')
        loaded_ids = traintest_ids + list(val_ids)
        print('Creating labels for dataset df')
        #print('All IDs:', loaded_ids)
        labels, cluster_go_proper_order = create_go_labels(loaded_ids, cluster_ann, 
            labels_to_ignore=set(to_remove))
        print('Labels shape:', labels.shape)
        
        # Create pure label DFs
        cluster_df = pl.DataFrame({
            'id': loaded_ids,
            'labels': labels
        })

        traintest_df = cluster_df.filter(pl.col('id').is_in(traintest_ids))
        id_to_fold = {}
        # Add fold info
        fold_ids = [traintest_ids_with_folds[pid] for pid in traintest_df['id'].to_list()]
        traintest_df = traintest_df.with_columns(fold_id=pl.Series(fold_ids))
        
        val_df = cluster_df.filter(pl.col('id').is_in(val_ids))
        
        print(len(traintest_df), 'proteins in cluster traintest df')
        print(len(val_df), 'proteins in cluster val df')

        traintest_folded, gos_to_remove = validate_folds(
            traintest_df, 
            n_folds, cluster_go_proper_order)

        if len(gos_to_remove) > 0:
            print('Removing GO terms which did not survive folding:', gos_to_remove)
            go_mask = [go not in gos_to_remove for go in cluster_go_proper_order]
            traintest_labels = traintest_folded['labels'].to_numpy()
            val_labels = val_df['labels'].to_numpy()
            traintest_labels_filtered_vec = [
                    np.array([val for should_use, val in zip(go_mask, prot_onehot_labels) if should_use])
                for prot_onehot_labels in traintest_labels]
            val_labels_filtered_vec = [
                    np.array([val for should_use, val in zip(go_mask, prot_onehot_labels) if should_use])
                for prot_onehot_labels in val_labels]
            traintest_folded = traintest_folded.with_columns(
                labels=np.asarray(traintest_labels_filtered_vec)
            )
            val_df = val_df.with_columns(
                labels=np.asarray(val_labels_filtered_vec)
            )
            cluster_go_proper_order = [go 
                for go, should_use in zip(cluster_go_proper_order, go_mask) if should_use]

        print('Calculating baseline metrics')
        val_df_y_np = val_df['labels'].to_numpy()
        rand_df_y_np = create_random_baseline(val_df['labels'].to_numpy())
        roc_auc_score_mac_base = metrics.roc_auc_score(val_df_y_np, rand_df_y_np, average='macro')
        roc_auc_score_w_base = metrics.roc_auc_score(val_df_y_np, rand_df_y_np, average='weighted')
        auprc_mac_base = metrics.average_precision_score(val_df_y_np, rand_df_y_np)
        auprc_w_base = metrics.average_precision_score(val_df_y_np, rand_df_y_np, average='weighted')

        #Save used and not used GO ids
        label_names_path = tmp_dir + '/labels_' + self.dataset_name + '-' + cluster_name + '.txt'
        open(label_names_path, 'w').write('\n'.join(cluster_go_proper_order))
        labels_not_used_path = tmp_dir + '/labels_not_used_' + self.dataset_name + '-' + cluster_name + '.txt'
        open(labels_not_used_path, 'w').write('\n'.join(set(to_remove).union(gos_to_remove)))

        print('Saving labels parquet to tmp')
        traintest_labels_folds_dir_tmp = tmp_dir + '/labels_traintest_folds' + str(n_folds) + '_' + self.dataset_name + '-' + cluster_name
        prepare_label_folds_obj(traintest_folded, n_folds, traintest_labels_folds_dir_tmp, folds_description)

        traintest_df_path = tmp_dir + '/labels_traintest_' + self.dataset_name + '-' + cluster_name + '.parquet'
        val_df_path = tmp_dir + '/labels_val_' + self.dataset_name + '-' + cluster_name + '.parquet'
        traintest_folded.write_parquet(traintest_df_path)
        val_df.write_parquet(val_df_path)
        del traintest_df
        del traintest_folded
        del val_df
        del cluster_df

        cluster = {
            'id': loaded_ids,
            'go': cluster_go_proper_order,
            'label_names_path': label_names_path,
            'labels_not_used_path': labels_not_used_path,
            'traintest_labels_path': traintest_df_path,
            'val_labels_path': val_df_path,
            'traintest_labels_folds_path': traintest_labels_folds_dir_tmp, # Store temp path to be moved later
            'baseline_metrics':{
                'roc_auc_score_mac': roc_auc_score_mac_base,
                'roc_auc_score_w': roc_auc_score_w_base,
                'auprc_mac': auprc_mac_base,
                'auprc_w': auprc_w_base
            }
        }
        
        return cluster

    def create_new_dataset(self, dimension_db: DimensionDB, min_proteins_per_mf: int, 
            dataset_type: str, val_perc: float):
        tmp_dir = 'local_tmp'
        if not path.exists(tmp_dir):
            mkdir(tmp_dir)
        
        self.release_n = dimension_db.release_dir.split("_")[-1]
        if 'cafa' in dimension_db.release_dir:
            self.release_n = path.basename(dimension_db.release_dir)
        self.min_proteins_per_mf = min_proteins_per_mf
        self.dataset_type = dataset_type
        self.dataset_name = dataset_type + '_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
        self.val_perc = val_perc

        self.dataset_params = {'release_n': self.release_n, 
            'min_ann': str(min_proteins_per_mf), 
            'dataset_type': dataset_type,
            'dataset_name': self.dataset_name,
            'val_perc': val_perc}
        traintest_set, val_set, filtered_ann, go_freqs = dimension_db.get_proteins_set(
            min_proteins_per_mf, val_perc)

        max_proteins_traintest = None

        print(self.dataset_params)
        if dataset_type in ['base_benchmark', 'taxon_benchmark', 'taxon_benchmark_cv']:
            print('base_benchmark_goids_clustering')
            go_clusters = base_benchmark_goids_clustering(dimension_db, go_freqs)
            max_proteins_traintest = 4500
            if dataset_type == 'base_benchmark':
                self.datasets_to_load = dimension_db.plm_names
            elif dataset_type == 'taxon_benchmark':
                self.datasets_to_load = ['ankh_base', 'prottrans'] + dimension_db.taxa_onehot_names + dimension_db.taxa_profile_names
            else:
                self.datasets_to_load = ['taxa_256', 'ankh_base', 'esm2_t33']
        elif dataset_type in ["full_swarm", 'small_swarm']:
            print('full_mf_goids_clustering')
            go_clusters = full_mf_goids_clustering(dimension_db, go_freqs, 
                len(traintest_set), is_test=False)
            if dataset_type == 'small_swarm':
                cluster_names_all = sorted(go_clusters.keys())
                print('All clusters: ', cluster_names_all)
                small_cluster_names = [cluster_names_all[0],
                    cluster_names_all[1]]
                print('Only using: ', small_cluster_names)
                go_clusters = {c_name: go_clusters[c_name] for c_name in small_cluster_names}
            self.datasets_to_load = ['taxa_256', 'ankh_base', 'esm2_t33']
        if 'cafa' in self.release_n:
            self.datasets_to_load = [dataset_name + '.train' 
                for dataset_name in self.datasets_to_load]
        self.dataset_params['datasets_to_load'] = self.datasets_to_load

        self.go_clusters = {}
        parquet_loader = VectorLoader(dimension_db.release_dir)
        
        print('Loading all vectors and assigning folds...')
        # Pre-load all vectors to define the universe of proteins
        # remove_na=True means checking availability in all datasets
        traintest_set = list(traintest_set)
        val_set = list(val_set)
        all_candidate_ids = traintest_set + val_set
        
        vectors_df = parquet_loader.load_vectors_by_ids(
            all_candidate_ids, 
            self.datasets_to_load,
            remove_na=True)
        
        available_ids = set(vectors_df['id'].to_list())
        all_ordered_ids = vectors_df['id'].to_list()
        
        # Filter sets based on availability AND enforce order from vectors_df
        traintest_set_set = set(traintest_set)
        val_set_set = set(val_set)
        
        traintest_set = [pid for pid in all_ordered_ids if pid in traintest_set_set]
        val_set = [pid for pid in all_ordered_ids if pid in val_set_set]
        
        # Assign folds to traintest set
        # We shuffle a copy to assign random folds, but keep the original list ordered
        traintest_for_folding = list(traintest_set)
        random.shuffle(traintest_for_folding)
        traintest_folds = np.array_split(traintest_for_folding, self.n_folds) # 5 folds constant
        
        traintest_ids_with_folds = {}
        for fold_idx, fold_ids in enumerate(traintest_folds):
            for pid in fold_ids:
                traintest_ids_with_folds[pid] = fold_idx
        
        # Create feature DFs with fold info
        # Filtering preserves order of vectors_df
        features_traintest_df = vectors_df.filter(pl.col('id').is_in(traintest_set))
        
        # Add fold info to features df
        fold_ids_col = [traintest_ids_with_folds[pid] for pid in features_traintest_df['id'].to_list()]
        features_traintest_df = features_traintest_df.with_columns(fold_id=pl.Series(fold_ids_col))
        
        features_val_df = vectors_df.filter(pl.col('id').is_in(val_set))
        
        print('Saving feature parquets to tmp...')
        self.features_traintest_path = tmp_dir + '/features_traintest_' + self.dataset_name + '.parquet'
        self.features_val_path = tmp_dir + '/features_val_' + self.dataset_name + '.parquet'
        
        features_traintest_df.write_parquet(self.features_traintest_path)
        features_val_df.write_parquet(self.features_val_path)

        # Generate global feature folds (generic, including all available features)
        # Include 'id' and all feature columns (but not fold_id) for validation purposes
        feature_cols = [c for c in features_traintest_df.columns if c != 'fold_id']
        self.features_folds_dir_tmp = tmp_dir + '/features_traintest_folds' + str(self.n_folds) + '_' + self.dataset_name
        _, folds_description = prepare_global_feature_folds(features_traintest_df, self.n_folds, 
            self.features_folds_dir_tmp, feature_cols)
        
        traintest_original_order = features_traintest_df['id'].to_list()

        del vectors_df
        del features_traintest_df
        del features_val_df

        print('Making datasets for clusters')

        for cluster_name, cluster_gos in tqdm(go_clusters.items(), total = len(go_clusters)):
            self.go_clusters[cluster_name] = self.goids_to_cluster(cluster_name, cluster_gos, filtered_ann, 
                traintest_original_order, traintest_ids_with_folds, val_set, tmp_dir, dimension_db, folds_description,
                max_proteins_traintest=max_proteins_traintest, n_folds=self.n_folds)

        for clustername in self.go_clusters.keys():
            print(clustername, 'cluster created')
        
        self.go_ids = sorted(set(concat_lists([g['go'] for g in self.go_clusters.values()])))
        all_ids = set(concat_lists([g['id'] for g in self.go_clusters.values()]))
        self.ids = [p for p in dimension_db.ids if p in all_ids]
        
        print(len(self.ids), 'proteins')
        print(len(self.go_ids), 'go ids')

    # Clustering methods removed (delegated to clustering.py)
    
    # Validation methods moved to validation.py
    def validate_dataset_integrity(self, datasets_dir, n_folds=5):
        """Validate all generated .obj files against their source .parquet files."""
        return validation.validate_dataset_integrity(self, datasets_dir, n_folds)
    
    def save_to_dir(self, outputdir):
        run_command(['mkdir -p', outputdir])
        json.dump(self.dataset_params, open(path.join(outputdir, 'params.json'), 'w'), indent=4)
        if path.exists(self.features_traintest_path):
             final_features_tt = outputdir + '/features_traintest.parquet'
             run_command(['mv', self.features_traintest_path, final_features_tt])
        
        if path.exists(self.features_val_path):
             final_features_val = outputdir + '/features_val.parquet'
             run_command(['mv', self.features_val_path, final_features_val])
        
        if path.exists(self.features_folds_dir_tmp):
             final_features_folds = outputdir + '/features_traintest_folds' + str(self.n_folds)
             if path.exists(final_features_folds):
                 run_command(['rm -rf', final_features_folds])
             run_command(['mv', self.features_folds_dir_tmp, final_features_folds])

        for cluster_name, cluster_dict in self.go_clusters.items():
            df_path1_tmp = cluster_dict['traintest_labels_path']
            df_path1 = outputdir + '/labels_traintest_' + cluster_name + '.parquet'
            run_command(['mv', df_path1_tmp, df_path1])
            cluster_dict['traintest_labels_path'] = df_path1

            df_path2_tmp = cluster_dict['val_labels_path']
            df_path2 = outputdir + '/labels_val_' + cluster_name + '.parquet'
            run_command(['mv', df_path2_tmp, df_path2])
            cluster_dict['val_labels_path'] = df_path2

            # Move pre-generated label folds
            folds_path_tmp = cluster_dict['traintest_labels_folds_path']
            final_folds_path = outputdir + '/labels_traintest_folds' + str(self.n_folds) + '_' + cluster_name
            if path.exists(final_folds_path):
                 run_command(['rm -rf', final_folds_path])
            run_command(['mv', folds_path_tmp, final_folds_path])
            
            del cluster_dict['traintest_labels_folds_path']
            
            expected_folds_path = df_path1.replace('.parquet', '_folds' + str(self.n_folds)) # Assuming n_folds=5 fixed here
            run_command(['mv', final_folds_path, expected_folds_path]) # Move to expected location

            #Save label names files
            run_command(['mv', cluster_dict['label_names_path'], outputdir])
            run_command(['mv', cluster_dict['labels_not_used_path'], outputdir])

        json.dump(self.go_clusters, 
            gzip.open(path.join(outputdir, 'go_clusters.json.gz'), 'wt'), indent=4)
        ids_str = '\n'.join(self.ids)
        open(path.join(outputdir, 'ids.txt'), 'w').write(ids_str)
        gos_str = '\n'.join(self.go_ids)
        open(path.join(outputdir, 'go_ids.txt'), 'w').write(gos_str)

    def save(self, datasets_dir):
        outputdir = path.join(datasets_dir, self.dataset_name)
        if not path.exists(outputdir):
            run_command(['mkdir -p', outputdir])
        self.save_to_dir(outputdir)

def find_or_create_dataset(datasets_dir, dataset_type, min_proteins_per_mf, 
        dimension_db_release_n, dimension_db_releases_dir, val_perc):
    matching_dataset_path = find_latest_dataset(datasets_dir, dataset_type, 
                                            min_proteins_per_mf, dimension_db_release_n,
                                            val_perc)
    dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, new_downloads=True)
    if matching_dataset_path is not None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dataset = Dataset(dimension_db=dimension_db, 
                      min_proteins_per_mf=min_proteins_per_mf, 
                      dataset_type=dataset_type,
                      val_perc=val_perc)
    print('Nodes in dataset:', dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)
    
    return dataset, dimension_db
