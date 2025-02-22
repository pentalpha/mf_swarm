from collections import Counter
import gzip
from os import mkdir, path, stat
import json
from tqdm import tqdm
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

from util_base import concat_lists, run_command, irrelevant_mfs

protein_dimension_db_url = 'https://ucrania.imd.ufrn.br/~pitagoras/protein_dimension_db'
#protein_dimension_db_url = '$UCRANIA_INTERNAL/~pitagoras/protein_dimension_db'

def load_annotation_terms(file_path: str):
    by_protein = {}
    for rawline in gzip.open(file_path, 'rt'):
        cells = rawline.rstrip('\n').split("\t")
        uniprot = cells[0]
        mfs = cells[1].split(',')
        mfs = [x for x in mfs if x.startswith("GO:")]
        by_protein[uniprot] = mfs

    return by_protein

class DimensionDB:
    def __init__(self, dimension_db_releases_dir, dimension_db_release_n, new_downloads = False) -> None:
        print('Creating file names')
        self.release_dir        = path.join(dimension_db_releases_dir, 'release_' + dimension_db_release_n)
        self.release_url        = protein_dimension_db_url + '/release_'+ dimension_db_release_n
        self.plm_names          = ['esm2_t6', 'esm2_t12', 'esm2_t30', 
                                   'ankh_base', 'prottrans', 'esm2_t33', 'ankh_large', 
                                   'esm2_t36']
        self.emb_extension      = '.npy.gz'
        self.new_emb_extension  = '.parquet'
        self.taxa_profile_names = ['taxa_profile_128', 'taxa_profile_256']
        self.taxa_onehot_names  = ['taxa_256', 'taxa_128']
        self.top_taxa_names     = ['top_taxa_128', 'top_taxa_256']

        run_command(['mkdir -p', self.release_dir])

        self.feature_paths = {}
        for n in self.taxa_onehot_names:
            self.feature_paths[n] = path.join(self.release_dir, 'onehot.' + n + self.emb_extension)
        for n in self.taxa_profile_names:
            self.feature_paths[n] = path.join(self.release_dir, 'emb.' + n + self.emb_extension)
        for n in self.plm_names:
            self.feature_paths[n] = path.join(self.release_dir, 'emb.' + n + self.new_emb_extension)
        for n in self.top_taxa_names:
            self.feature_paths[n] = path.join(self.release_dir, n + '.txt')

        self.fasta_path = path.join(self.release_dir, 'uniprot_sorted.not_large.fasta')
        self.taxids_path = path.join(self.release_dir, 'taxid.tsv')
        self.ids_uniprot_path = path.join(self.release_dir, 'ids.txt')
        self.mf_gos_path = path.join(self.release_dir, 'go.experimental.mf.tsv.gz')
        self.go_basic_path = path.join(self.release_dir, 'go-basic.obo')
        self.other_files = [self.fasta_path, self.taxids_path, self.ids_uniprot_path, 
                            self.mf_gos_path, self.go_basic_path]

        if new_downloads:
            print('Checking for downloads')
            for p in tqdm(list(self.feature_paths.values()) + self.other_files):
                basename = path.basename(p)
                download_url = self.release_url + '/' + basename
                download_it = not path.exists(p)
                if not download_it:
                    gzip_finished = run_command(['gzip -t', p], no_output=True) == 0 if p.endswith('.gz') else True
                    if not gzip_finished or stat(p).st_size == 0:
                        download_it = True

                if download_it:
                    print("Downloading", p)
                    cmd = ['cd', self.release_dir, '&&', 'wget', '-O', basename, download_url, '2> /dev/null']
                    print(' '.join(cmd))
                    run_command(cmd)
                    if stat(p).st_size == 0:
                        run_command(['rm', p])
                else:
                    print('Not downloading', p)

        print("Checking for missing files")
        for name in self.feature_paths.keys():
            if not path.exists(self.feature_paths[name]):
                print(self.feature_paths[name], 'does not exist')
                self.feature_paths[name] = None
        
        print('Available features:')
        print(json.dumps(self.feature_paths, indent=4))

        for i in range(len(self.other_files)):
            p = self.other_files[i]
            print(p, path.exists(p))
            if not path.exists(p):
                self.other_files[i] = None

        print("Loading ids")
        self.taxids = open(self.taxids_path, 'r').read().split('\n')
        self.taxids = [l.split("\t")[1] for l in self.taxids]
        self.ids = open(self.ids_uniprot_path, 'r').read().split('\n')

        self.missing_features_path = self.release_dir + '/missing_features.json'
        if not path.exists(self.missing_features_path):
            self.list_missing_features()
        self.missing_features = json.load(open(self.missing_features_path, 'r'))

        self.validation_splits_dir = self.release_dir + '/validation_splits'
        if not path.exists(self.validation_splits_dir):
            mkdir(self.validation_splits_dir)

    def list_missing_features(self):
        missing_dict = {x: [] for x in self.ids}

        for feature_name, file_path in self.feature_paths.items():
            if file_path:
                if '.parquet' in file_path and 'emb.' in file_path:
                    print('Looking for NaN s in ', file_path)
                    parquet_file = pq.ParquetFile(file_path)
                    n_nan = 0
                    for batch in tqdm(parquet_file.iter_batches()):
                        new_df = batch.to_pandas()
                        if 'emb' in new_df:
                            loaded_embs = new_df['emb'].to_list()
                            local_ids = new_df['id'].to_list()
                            for protid, emb in zip(local_ids, loaded_embs):
                                if np.isnan(emb).any():
                                    missing_dict[protid].append(feature_name)
                                    n_nan += 1
                        else:
                            print('No "emb" column in', file_path, ', ignoring')
                            break
                    print(n_nan, 'nan values found')
            else:
                print(file_path, 'not found')
        
        json.dump(missing_dict, open(self.missing_features_path, 'w'), indent=4)

    def list_usable_proteins(self, min_proteins_per_mf):
        print(len(self.ids), 'proteins')
        all_swissprot_ids = set([k for k, v in self.missing_features.items() if len(v) == 0])
        print(len(all_swissprot_ids), 'proteins with all features present')

        all_annotations = load_annotation_terms(file_path=self.mf_gos_path)
        all_proteins = set(all_annotations.keys())
        print(len(all_proteins), 'proteins in', self.mf_gos_path)
        all_goids = set(concat_lists([list(v) for v in all_annotations.values()]))
        print(len(all_goids), 'GO IDs in', self.mf_gos_path)

        all_annotations = {k: set(v) for k, v in all_annotations.items() if k in all_swissprot_ids}
        all_proteins = set(all_annotations.keys())
        print(len(all_proteins), 'uniprot proteins with features in', self.mf_gos_path)
        goid_list = concat_lists([list(v) for v in all_annotations.values()])
        goid_list = [goid for goid in goid_list if not goid in irrelevant_mfs]
        print(len(set(goid_list)), 'GO IDs from uniprot proteins in', self.mf_gos_path)

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
        print(len(frequent_go_ids), 'frequent GO IDs from uniprot proteins in', self.mf_gos_path)
        
        all_annotations = {k: v.intersection(frequent_go_ids) for k, v in all_annotations.items()}
        all_annotations = {k: v for k, v in all_annotations.items() if len(v) > 0}
        annotated_proteins = [k for k in self.ids if k in all_annotations]
        go_freqs_filtered = {g: c for g, c in go_counts.items() if g in frequent_go_ids}
        print(len(annotated_proteins), 'proteins annotated to frequent GO IDs')

        return annotated_proteins, all_annotations, go_freqs_filtered
    
    def split_by_taxid(proteins_by_taxid, val_perc, min_proteins_in_taxid):
        val_set = set()
        traintest_set = set()

        unfrequent_taxid_proteins = set()

        #print('Creating random validation splits by taxon id')
        for taxid, proteins in proteins_by_taxid.items():
            if len(proteins) < min_proteins_in_taxid:
                unfrequent_taxid_proteins.update(proteins)
            else:
                #val_n = int(round(len(proteins)*val_perc))
                ids_train, ids_validation = train_test_split(
                    proteins, test_size = val_perc)
                val_set.update(ids_validation)
                traintest_set.update(ids_train)
        
        #print('Splitting', len(unfrequent_taxid_proteins), 'proteins in unfrequent taxa')
        ids_train, ids_validation = train_test_split(list(unfrequent_taxid_proteins), 
            test_size = val_perc)
        val_set.update(ids_validation)
        traintest_set.update(ids_train)

        return traintest_set, val_set

    def sep_validation(self, proteins, val_perc, annotations):
        print('Validation percent:', val_perc)
        print('Grouping', len(proteins) , 'proteins by species')
        proteins_by_taxid = {taxid: [] for taxid in set(self.taxids)}
        proteins_set = set(proteins)
        for i in tqdm(range(len(self.ids)), total = len(self.ids)):
            if self.ids[i] in proteins_set:
                proteins_by_taxid[self.taxids[i]].append(self.ids[i])
        min_proteins_in_taxid = int(round(100 / (val_perc*100)))
        print('Min proteins for taxid specific sampling:', min_proteins_in_taxid)
        
        print('Indexing proteins by go id')
        ann_by_go = {}
        for protid, prot_ann in tqdm(annotations.items()):
            for goid in prot_ann:
                if not goid in ann_by_go:
                    ann_by_go[goid] = set([protid])
                else:
                    ann_by_go[goid].add(protid)

        print('Trying different splits')
        splits = []
        min_so_far = 300
        for _ in tqdm(range(600)):
            traintest_set, val_set = DimensionDB.split_by_taxid(proteins_by_taxid, val_perc, 
                                                                min_proteins_in_taxid)
            faulty_go_ids = set()

            for goid, prot_ann in ann_by_go.items():
                local_traintest = prot_ann.intersection(traintest_set)
                local_val = prot_ann.intersection(val_set)
                if len(local_traintest) <= 3:
                    #print(goid, 'has zero samples in traintest set')
                    faulty_go_ids.add(goid)
                elif len(local_val) <= 3:
                    #print(goid, 'has zero samples in validation set')
                    faulty_go_ids.add(goid)
            if len(faulty_go_ids) < min_so_far:
                min_so_far = len(faulty_go_ids)
                print('Split with', len(faulty_go_ids), 'faulty GO ids')
            splits.append({'faulty_go_ids': faulty_go_ids, 
                           'traintest_set': traintest_set, 'val_set': val_set})
        
        splits.sort(key=lambda s: len(s['faulty_go_ids']))
        best_split = splits[0]
        traintest_set = best_split['traintest_set']
        val_set = best_split['val_set']

        val_sorted = []
        traintest_sorted = []
        for protein_id in self.ids:
            if protein_id in val_set:
                val_sorted.append(protein_id)
            elif protein_id in traintest_set:
                traintest_sorted.append(protein_id)
        
        print('Split:', len(traintest_sorted), len(val_sorted))
        
        return traintest_sorted, val_sorted, faulty_go_ids
    
    def get_proteins_set(self, min_proteins_per_mf: int, val_perc: float):
        proteins_set_dir = self.validation_splits_dir + '/min_prot' + str(min_proteins_per_mf) + '_val' + str(val_perc)
        validation_path = proteins_set_dir + '/validation_ids.txt'
        traintest_path = proteins_set_dir + '/traintest_ids.txt'
        annotations_path = proteins_set_dir+'/annotations.json'
        go_freqs_path = proteins_set_dir+'/go_freqs.json'
        paths = [proteins_set_dir, validation_path, traintest_path, annotations_path, go_freqs_path]
        if not all([path.exists(x) for x in paths]):
            proteins, annotations, go_freqs = self.list_usable_proteins(min_proteins_per_mf)
            traintest, validation, faulty_go_ids = self.sep_validation(proteins, val_perc, annotations)
            for goid in faulty_go_ids:
                del go_freqs[goid]
            print('Removing faulty GO ids')
            for prot in annotations.keys():
                to_remove = annotations[prot].intersection(faulty_go_ids)
                if len(to_remove) > 0:
                    for t in to_remove:
                        annotations[prot].remove(t)
            annotations = {k: list(v) for k, v in annotations.items()}
            print('Saving split')
            run_command(['mkdir -p', proteins_set_dir])
            open(validation_path, 'w').write('\n'.join(validation))
            open(traintest_path, 'w').write('\n'.join(traintest))
            json.dump(annotations, open(annotations_path, 'w'))
            json.dump(go_freqs, open(go_freqs_path, 'w'))
        
        validation = open(validation_path, 'r').read().split('\n')
        traintest = open(traintest_path, 'r').read().split('\n')
        annotations = json.load(open(annotations_path, 'r'))
        go_freqs = json.load(open(go_freqs_path, 'r'))

        annotations = {k: set(v) for k, v in annotations.items()}
        
        return traintest, validation, annotations, go_freqs