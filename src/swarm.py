from glob import glob
import json
from os import path, mkdir
from pickle import load, dump
from typing import List

import obonet
import networkx as nx
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

from create_dataset import Dataset
from custom_statistics import eval_predictions_dataset
from dimension_db import DimensionDB
from parquet_loading import VectorLoader
from util_base import run_command


class Node():
    def __init__(self, node_dir):
        self.node_dir = node_dir
        self.name = path.basename(node_dir)
        results_path = node_dir+'/standard_results.json'
        params_path = node_dir+'/standard_params.json'
        self.params = json.load(open(params_path, 'r'))
        self.results = json.load(open(results_path, 'r'))
        self.model_path = node_dir+'/standard_model.obj'
        self.validation1_results_path = node_dir + '/standard_validation.parquet'
        self.protein_labels_list = self.params['node']['go']
        self.basic_ensemble = None

    def load_model(self):
        self.basic_ensemble = load(open(self.model_path, 'rb'))
    
    def unload_model(self):
        del self.basic_ensemble

    def predict_scores(self, feature_lists, autounload=True):
        if self.basic_ensemble is None:
            self.load_model()
        base_model_results = [m.predict(feature_lists) 
            for m in self.basic_ensemble.models]
        results_mean = np.mean(base_model_results, axis=0)
        if autounload:
            self.unload_model()
        score_lists = base_model_results + [results_mean]
        score_names = [f'fold{f}_base' for f in range(len(base_model_results))] + ['mean']
        schema = {}
        for name, l in zip(score_names, score_lists):
            schema[name] = l
        scores_df = pl.DataFrame(schema)
        return scores_df

def join_protein_score_dicts(protein_scores: dict, go_sequence: list, has_annots: bool = False):
    ids = []
    all_scores = []
    all_annots = []
    print('Joining information')
    for uniprot, scores_dict in tqdm(protein_scores.items(), total=len(protein_scores.keys())):
        if has_annots:
            scores_vec = [scores_dict[go][0] if go in scores_dict else 0.0 for go in go_sequence]
            annots_vec = [scores_dict[go][1] if go in scores_dict else 0.0 for go in go_sequence]
            all_annots.append(np.array(annots_vec))
        else:
            scores_vec = [scores_dict[go] if go in scores_dict else 0.0 for go in go_sequence]
        ids.append(uniprot)
        all_scores.append(np.array(scores_vec))
    
    recipe = { 'id': ids,
        'scores': np.asarray(all_scores)}
    if has_annots:
        recipe['labels'] = np.asarray(all_annots)

    df = pl.DataFrame(recipe)
    return df

def join_prediction_dfs_no_nodes(df_paths: List[str], go_lists: List[str], scores_col = 'mean'):
    all_gos = set()
    for go_list in go_lists:
        all_gos.update(go_list)
    final_go_seq = sorted(all_gos)
    
    protein_scores = {}
    for df_path, go_list in zip(df_paths, go_lists):
        df = pl.read_parquet(df_path)
        ids = df['id'].to_list()
        score_lists = df[scores_col].to_list()
        for uniprot, scores in zip(ids, score_lists):
            if not uniprot in protein_scores:
                protein_scores[uniprot] = {}
            
            for score, local_go in zip(scores, go_list):
                protein_scores[uniprot][local_go] = [score]
    print('Create final dataframe')
    df = join_protein_score_dicts(protein_scores, final_go_seq, has_annots=False)
    return df, final_go_seq

def join_prediction_datasets(nodes: List[Node]):
    all_gos = set()
    go_lists = []
    for node in nodes:
        go_lists.append(node.protein_labels_list)
        all_gos.update(node.protein_labels_list)
    print(len(all_gos))
    final_go_seq = sorted(all_gos)

    protein_scores = {}
    print('Loading node validation dfs')
    for node in tqdm(nodes):
        local_go_list = node.protein_labels_list
        val_df = pl.read_parquet(node.validation1_results_path)
        ids = val_df['id'].to_list()
        true_annots = val_df['y'].to_list()
        score_lists = val_df['y_pred'].to_list()
        for uniprot, scores, protein_annots in zip(ids, score_lists, true_annots):
            if not uniprot in protein_scores:
                protein_scores[uniprot] = {}
            
            for score, local_go, true_value in zip(scores, local_go_list, protein_annots):
                protein_scores[uniprot][local_go] = (score, true_value)
    
    print('Create final dataframe')
    df = join_protein_score_dicts(protein_scores, final_go_seq, has_annots=True)

    return df, final_go_seq

def go_descendants_dict(go_id_sequence, go_network):
    go_descendants = {}
    go_descendants_indexes = {}
    go_ids_set = set(go_network)
    for go_id in go_ids_set:
        #the go ontology points from child to parent
        descendants = [g for g in list(nx.ancestors(go_network, go_id)) if g in go_ids_set]
        go_descendants[go_id] = descendants
        go_descendants_indexes[go_id] = [n for n, g in enumerate(go_id_sequence) if g in descendants]
    return go_descendants, go_descendants_indexes

def go_ancestors_dict(go_id_sequence, go_network):
    go_ancestors = {}
    go_ancestors_indexes = {}
    go_ids_set = set(go_network)
    for go_id in go_id_sequence:
        #the go ontology points from child to parent
        ancestors = [g for g in list(nx.descendants(go_network, go_id)) if g in go_ids_set]
        go_ancestors[go_id] = ancestors
        go_ancestors_indexes[go_id] = [n for n, g in enumerate(go_id_sequence) if g in ancestors]
    return go_ancestors, go_ancestors_indexes

class Swarm:
    def __init__(self, nodes_dir: str, go_network_path: str):
        assert path.exists(nodes_dir)
        self.swarm_dir = nodes_dir
        self.configs = json.load(open(nodes_dir+'/configs_used.json', 'r'))
        self.nodes = [Node(node_dir) for node_dir in glob(nodes_dir + '/Level-*')]
        self.features = self.nodes[0].params['features']

        self.swarm_go_ids_path = self.swarm_dir + '/go_ids.txt'
        self.raw_metrics_path = self.swarm_dir + '/validation_results-raw_scores.txt'
        self.raw_scores_path = self.swarm_dir + '/validation-raw_scores.parquet'
        self.child_max_metrics_path = self.swarm_dir + '/validation_results-child_max_scores.txt'

        self.predictions_traintest_dir = self.swarm_dir + '/predictions_traintest'
        self.predictions_val_dir = self.swarm_dir + '/predictions_val'
        
        if any([not path.exists(x) for x in [self.swarm_go_ids_path, self.raw_metrics_path, self.raw_scores_path]]):
            self.analyze_raw_scores()
            self.raw_metrics = json.load(open(self.raw_metrics_path))
        else:
            self.raw_metrics = json.load(open(self.raw_metrics_path))
            print(json.dumps(self.raw_metrics, indent=2))
        self.go_id_sequence = open(self.swarm_go_ids_path).read().split('\n')
        
        print('Loading GO network')
        go_network = obonet.read_obo(go_network_path)
        self.go_descendants, self.go_descendants_indexes = go_descendants_dict(self.go_id_sequence, go_network)
        self.go_ancestors, self.go_ancestors_indexes = go_ancestors_dict(self.go_id_sequence, go_network)
        #Get distances to root:
        self.go_distances_to_root = {}
        root = 'GO:0003674'
        for go_id in self.go_id_sequence:
            self.go_distances_to_root[go_id] = nx.shortest_path_length(go_network, go_id, target=root)

        if not path.exists(self.child_max_metrics_path):
            self.calc_child_max_scores()
        self.child_max_metrics = json.load(open(self.child_max_metrics_path))
        self.metrics_df = self.make_metrics_df()
        self.metrics_df.write_csv(self.swarm_dir + '/metrics.csv')
            
    def analyze_raw_scores(self):
        raw_df, self.go_id_sequence = join_prediction_datasets(self.nodes)
        print(raw_df)
        raw_metrics = eval_predictions_dataset(raw_df)
        print('RAW Metrics:')
        print(json.dumps(raw_metrics, indent=2))
        json.dump(raw_metrics, open(self.raw_metrics_path, 'w'))
        open(self.swarm_go_ids_path, 'w').write('\n'.join(self.go_id_sequence))
        raw_df.write_parquet(self.raw_scores_path)

    def normalized_scores_child_max(self, scores_matrix: np.ndarray):
        norm_lines = []
        for i in tqdm(range(scores_matrix.shape[0])):
            prot_scores = scores_matrix[i]
            new_prot_scores = []
            for index, go_id in enumerate(self.go_id_sequence):
                descendants_indexes = self.go_descendants_indexes[go_id]
                if len(descendants_indexes) > 0:
                    new_score = max(prot_scores[descendants_indexes])
                else:
                    new_score = prot_scores[index]
                new_prot_scores.append(new_score)
            norm_lines.append(np.array(new_prot_scores))
        return np.asarray(norm_lines)
    
    def calc_child_max_scores(self):
        raw_df = pl.read_parquet(self.raw_scores_path)
        scores_matrix = raw_df['scores'].to_numpy()
        print('Normalizing scores')
        norm_scores_matrix = self.normalized_scores_child_max(scores_matrix)
        print('Evaluating child max scores')
        raw_df = raw_df.with_columns(pl.Series('scores', norm_scores_matrix))
        child_max_metrics = eval_predictions_dataset(raw_df)
        print('Child Max Metrics:')
        print(json.dumps(child_max_metrics, indent=2))
        json.dump(child_max_metrics, open(self.child_max_metrics_path, 'w'))

    def make_metrics_df(self):
        metrics_list = ["ROC AUC", "ROC AUC", "AUPRC", "AUPRC W", "Fmax", "Best F1 Threshold"]
        metrics_dicts = [('Raw Scores', self.raw_metrics), ('Child Max Norm.', self.child_max_metrics)]
        lines = []
        for name, d in metrics_dicts:
            line = {'Name': 'MF Swarm '+name}
            for metric in metrics_list:
                line[metric] = d[metric]
            lines.append(line)
        df = pl.DataFrame(lines)
        print(df)
        return df

    def predict_on_dataset(self, features_df: pl.DataFrame, output_dir: str,
            autounload=True):
        feature_lists = [features_df[col].to_numpy() for col in self.features]
        if not path.exists(output_dir):
            mkdir(output_dir)
        df_paths = []
        go_lists = []
        for node in tqdm(self.nodes):
            df_path = output_dir + f'/{node.name}-scores.parquet'
            go_list = output_dir + f'/{node.name}-go_ids.txt'

            if not path.exists(df_path):
                scores_df = node.predict_scores(feature_lists, autounload=autounload)
                scores_df = scores_df.with_columns(pl.Series('id', features_df['id'].to_list()))
                scores_df.write_parquet(df_path)
                open(go_list, 'w').write('\n'.join(node.protein_labels_list))
            df_paths.append(df_path)
            go_lists.append(node.protein_labels_list)

        df, final_go_seq = join_prediction_dfs_no_nodes(df_paths, go_lists)
        df.write_parquet(output_dir + '/mean_scores.parquet')
        open(output_dir + '/go_ids.txt', 'w').write('\n'.join(final_go_seq))
        return df, final_go_seq

    def make_all_predictions(self, dimension_db: DimensionDB):
        traintest_set, val_set, filtered_ann, go_freqs = dimension_db.get_proteins_set(
            self.configs['min_proteins_per_mf'], self.configs['val_perc'])
        
        parquet_loader = VectorLoader(dimension_db.release_dir)
        run_command(['mkdir', '-p', self.predictions_traintest_dir, self.predictions_val_dir])
        traintest_parquet_path = self.predictions_traintest_dir + '/mean_scores.parquet'
        val_parquet_path = self.predictions_val_dir + '/mean_scores.parquet'
        traintest_list = list(traintest_set)
        val_list = list(val_set)

        features_df_path = traintest_parquet_path.replace('mean_scores.parquet', 'features.parquet')
        if not path.exists(features_df_path):
            features_df = parquet_loader.load_vectors_by_ids(traintest_list, self.features,
                remove_na=True)
            features_df.write_parquet(features_df_path)
        else:
            features_df = pl.read_parquet(features_df_path)
        
        if not path.exists(traintest_parquet_path):
            df, go_final_seq = self.predict_on_dataset(features_df, self.predictions_traintest_dir)
            df.write_parquet(traintest_parquet_path)
            open(self.predictions_traintest_dir + '/go_ids.txt', 'w').write('\n'.join(go_final_seq))
            print('Traintest predictions saved')
        else:
            df = pl.read_parquet(traintest_parquet_path)
            go_final_seq = open(self.predictions_traintest_dir + '/go_ids.txt').read().split('\n')
        
        val_features_df_path = val_parquet_path.replace('mean_scores.parquet', 'features.parquet')
        if not path.exists(val_features_df_path):
            val_features_df = parquet_loader.load_vectors_by_ids(val_list, self.features,
                remove_na=True)
            val_features_df.write_parquet(val_features_df_path)
        else:
            val_features_df = pl.read_parquet(val_features_df_path)
        
        if not path.exists(val_parquet_path):
            val_df, val_go_final_seq = self.predict_on_dataset(val_features_df, self.predictions_val_dir)
            val_df.write_parquet(val_parquet_path)
            open(self.predictions_val_dir + '/go_ids.txt', 'w').write('\n'.join(val_go_final_seq))
            print('Val predictions saved')
        else:
            val_df = pl.read_parquet(val_parquet_path)
            val_go_final_seq = open(self.predictions_val_dir + '/go_ids.txt').read().split('\n')
        
        return df, go_final_seq, val_df, val_go_final_seq
    
    def train_normalization(self, n_proteins: int = 5000, scores_per_line: int = 100):
        raw_df = pl.read_parquet(self.raw_scores_path)
        scores_matrix = raw_df['scores'].to_numpy()
        labels_matrix = raw_df['labels'].to_numpy()
        protein_indexes = np.random.choice(scores_matrix.shape[0], size=n_proteins, replace=False)
        scores_matrix = scores_matrix[protein_indexes]
        
        print('Creating features and labels')
        feature_lines = []
        feature_names = ['dist_to_root', 'current_score', 'top_min', 'top_q1', 'top_q2', 'top_q3', 'top_max',
            'bottom_min', 'bottom_q1', 'bottom_q2', 'bottom_q3', 'bottom_max']
        correct_scores = []
        for i in tqdm(range(scores_matrix.shape[0])):
            prot_scores = scores_matrix[i]
            correct_preds = labels_matrix[i]
            indexes_to_make_features = np.random.choice(len(self.go_id_sequence), size=scores_per_line, replace=False)
            for go_id_index in indexes_to_make_features:
                go_id = self.go_id_sequence[go_id_index]
                top_indexes = self.go_descendants_indexes[go_id]
                bottom_indexes = self.go_ancestors_indexes[go_id]
                top_scores = prot_scores[top_indexes]
                bottom_scores = prot_scores[bottom_indexes]
                if len(top_scores) > 0:
                    top_features = [np.min(top_scores), np.quantile(top_scores, 0.25), np.quantile(top_scores, 0.5), np.quantile(top_scores, 0.75), np.max(top_scores)]
                else:
                    top_features = [0, 0, 0, 0, 0]
                if len(bottom_scores) > 0:
                    bottom_features = [np.min(bottom_scores), np.quantile(bottom_scores, 0.25), np.quantile(bottom_scores, 0.5), np.quantile(bottom_scores, 0.75), np.max(bottom_scores)]
                else:
                    bottom_features = [0, 0, 0, 0, 0]
                feature_line = [self.go_distances_to_root[go_id], prot_scores[go_id_index]] + top_features + bottom_features
                feature_lines.append(np.array(feature_line))
                correct_scores.append(correct_preds[go_id_index])

        feature_lines = np.asarray(feature_lines)
        correct_scores = np.asarray(correct_scores)
        print('Features matrix shape:', feature_lines.shape)
        print('Labels matrix shape:', correct_scores.shape)

        print('Split data and train RandomForestRegressor')
        X_train, X_test, y_train, y_test = train_test_split(feature_lines, correct_scores, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print('Test R^2:', r2_score(y_test, y_pred))