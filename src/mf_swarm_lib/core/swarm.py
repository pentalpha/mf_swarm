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

from mf_swarm_lib.core.ml.custom_statistics import (calc_deepred_scores, eval_predictions_dataset, 
    eval_predictions_dataset_bool, find_best_threshold_per_col)
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.data.parquet_loading import VectorLoader
from mf_swarm_lib.utils.util_base import run_command
from mf_swarm_lib.core.node import (Node, go_descendants_dict, go_ancestors_dict, all_paths_to_root, 
    join_prediction_datasets, join_prediction_dfs_no_nodes)

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
        self.parents_min_metrics_path = self.swarm_dir + '/validation_results-parents_min_scores.txt'

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
        
        self.paths_to_root_path = self.swarm_dir + '/paths_to_root.json'
        if not path.exists(self.paths_to_root_path):
            self.go_paths_to_root = all_paths_to_root(self.go_id_sequence, go_network, root)
            json.dump(self.go_paths_to_root, open(self.paths_to_root_path, 'w'), indent=2)
        self.paths_to_root = json.load(open(self.paths_to_root_path, 'r'))

        if not path.exists(self.child_max_metrics_path):
            self.calc_simple_hierarchical_scores(norm_method='child_max')
        if not path.exists(self.parents_min_metrics_path):
            self.calc_simple_hierarchical_scores(norm_method='parents_min')
        self.child_max_metrics = json.load(open(self.child_max_metrics_path))
        self.parents_min_metrics = json.load(open(self.parents_min_metrics_path))

        self.go_id_thresholds_path = self.swarm_dir + '/go_id_thresholds.json'
        if not path.exists(self.go_id_thresholds_path):
            raw_df = pl.read_parquet(self.raw_scores_path)
            self.go_id_thresholds = find_best_threshold_per_col(
                raw_df['scores'].to_numpy(), raw_df['labels'].to_numpy(), 
                self.go_id_sequence)
            json.dump(self.go_id_thresholds, open(self.go_id_thresholds_path, 'w'), indent=2)
        self.go_id_thresholds = json.load(open(self.go_id_thresholds_path, 'r'))

        self.raw_metrics_by_label_path = self.swarm_dir + '/raw_metrics_by_label.json'
        if not path.exists(self.raw_metrics_by_label_path):
            raw_metrics_best_by_label = eval_predictions_dataset(raw_df,
                go_id_sequence=self.go_id_sequence,
                thresholds=self.go_id_thresholds)
            json.dump(raw_metrics_best_by_label, open(self.raw_metrics_by_label_path, 'w'), indent=2)
        self.raw_metrics_best_by_label = json.load(open(self.raw_metrics_by_label_path, 'r'))
        print('Raw metrics by label:')
        print(json.dumps(self.raw_metrics_best_by_label, indent=2))

        self.deepred_norm_metrics_path = self.swarm_dir + '/validation_results-deepred.txt'
        if not path.exists(self.deepred_norm_metrics_path):
            self.deepred_metrics = self.evaluate_deepred_normalization()
            json.dump(self.deepred_metrics, open(self.deepred_norm_metrics_path, 'w'), indent=2)
        self.deepred_metrics = json.load(open(self.deepred_norm_metrics_path, 'r'))

        '''self.deepred_norm2_metrics_path = self.swarm_dir + '/validation_results-deepred2.txt'
        if not path.exists(self.deepred_norm2_metrics_path):
            self.deepred_metrics2 = self.evaluate_deepred_normalization2()
            json.dump(self.deepred_metrics2, open(self.deepred_norm2_metrics_path, 'w'), indent=2)
        self.deepred_metrics2 = json.load(open(self.deepred_norm2_metrics_path, 'r'))'''

        self.metrics_df = self.make_metrics_df()
        self.metrics_df = self.metrics_df.sort(by=['Curve Fmax', 'Discrete F1 W', 'Discrete Precision W'])
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
    
    def normalized_scores_parents_min(self, scores_matrix: np.ndarray):
        norm_lines = []
        for i in tqdm(range(scores_matrix.shape[0])):
            prot_scores = scores_matrix[i]
            new_prot_scores = []
            for index, go_id in enumerate(self.go_id_sequence):
                ancestors_indexes = self.go_ancestors_indexes[go_id]
                if len(ancestors_indexes) > 0:
                    new_score = min(prot_scores[ancestors_indexes])
                else:
                    new_score = prot_scores[index]
                new_prot_scores.append(new_score)
            norm_lines.append(np.array(new_prot_scores))
        return np.asarray(norm_lines)
    
    def calc_simple_hierarchical_scores(self, norm_method='child_max'):
        raw_df = pl.read_parquet(self.raw_scores_path)
        scores_matrix = raw_df['scores'].to_numpy()
        print('Normalizing scores')
        if norm_method == 'child_max':
            norm_scores_matrix = self.normalized_scores_child_max(scores_matrix)
        elif norm_method == 'parents_min':
            norm_scores_matrix = self.normalized_scores_parents_min(scores_matrix)
        print('Evaluating child max scores')
        raw_df = raw_df.with_columns(pl.Series('scores', norm_scores_matrix))
        child_max_metrics = eval_predictions_dataset(raw_df)
        print('Child Max Metrics:')
        print(json.dumps(child_max_metrics, indent=2))
        child_max_metrics['norm_method'] = norm_method
        if norm_method == 'child_max':
            save_path = self.child_max_metrics_path
        elif norm_method == 'parents_min':
            save_path = self.parents_min_metrics_path
        json.dump(child_max_metrics, open(save_path, 'w'))

    def evaluate_deepred_normalization(self):
        fmax_value = self.raw_metrics['Best F1 Threshold']
        print('Fmax th value:', fmax_value)
        '''th_diffs = [0.0, -0.175, -0.15, -0.125, 
                    -0.1, -0.075, -0.05, -0.025, 
                    0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
        alternative_thresholds = [fmax_value+x for x in th_diffs]
        alternative_thresholds = [x for x in alternative_thresholds if x > 0 and x < 1]'''
        raw_df = pl.read_parquet(self.raw_scores_path)
        results = {}
        results['raw_fmax_th'] = eval_predictions_dataset_bool(
            raw_df['scores'].to_numpy() > fmax_value,
            raw_df['labels'].to_numpy() > 0
        )
        self.raw_metrics['boolean'] = results['raw_fmax_th']
        print('Raw Fmax Threshold:', json.dumps(results['raw_fmax_th'], indent=2))
        '''alternative_thresholds = [self.go_id_thresholds] + alternative_thresholds
        for threshold in alternative_thresholds:
            deepred_metrics = calc_deepred_scores(raw_df, self.go_id_sequence, 
                                                  self.paths_to_root, 
                                                  threshold=threshold)
            th_id = 'DeePred Norm ' + ('Min '+str(round(threshold, 3)) if isinstance(threshold, float) else 'By GO ID')
            results[th_id] = deepred_metrics'''
        deepred_metrics = calc_deepred_scores(raw_df, self.go_id_sequence, 
            self.paths_to_root, 
            threshold=self.go_id_thresholds)
        th_id = 'DeePred Norm By GO ID'
        results[th_id] = deepred_metrics
        return results
    
    def evaluate_deepred_normalization2(self):
        raw_df = pl.read_parquet(self.raw_scores_path)
        results = {}

        positive_proportions = [
            0.25, 0.275,
            0.3, 0.325, 0.35, 0.375, 
            0.4, 0.425, 0.45, 0.475, 
            0.525, 0.55, 0.575,
            0.6, 0.625, 0.65, 0.675,
            0.7, 0.75]
        for x in positive_proportions:
            deepred_metrics = calc_deepred_scores(raw_df, self.go_id_sequence, 
                self.paths_to_root, 
                threshold=self.go_id_thresholds,
                min_prop= x)
            th_id = 'DeePred Norm Min Prop ' + str(round(x, 3))
            results[th_id] = deepred_metrics
        return results

    def make_metrics_df(self):
        curve_metrics = ["ROC AUC", "ROC AUC W", "AUPRC", "AUPRC W", "Fmax", "Best F1 Threshold"]
        bool_metrics = ["F1 W", "Precision W", "Recall W", 
                        "Fmax", "ROC AUC", "ROC AUC W", "AUPRC", "AUPRC W"]
        
        self.raw_metrics['boolean'] = self.deepred_metrics['raw_fmax_th']

        metrics_dicts = [('Raw Scores', self.raw_metrics), 
                         ('Raw Scores Best by Label', self.raw_metrics_best_by_label),
                         ('Child Max Norm.', self.child_max_metrics), 
                         ('Parents Min Norm.', self.parents_min_metrics)]
        for name, d in self.deepred_metrics.items():
            metrics_dicts.append((name, d))
        '''for name, d in self.deepred_metrics2.items():
            metrics_dicts.append((name, d))'''
        
        lines = []
        for name, d in metrics_dicts:
            print('Metrics for', name)
            print(d)
            line = {'Name': 'MF Swarm '+name}
            if 'F1 W' in d and 'boolean' not in d:
                #boolean metrics only
                main_metrics_dict = self.raw_metrics
                for metric in curve_metrics:
                    line['Curve '+metric] = main_metrics_dict[metric]
                for metric in bool_metrics:
                    line['Discrete '+metric] = d[metric]
            else:
                #complete metrics set
                for metric in curve_metrics:
                    line['Curve '+metric] = d[metric]
                
                for metric in bool_metrics:
                    if 'boolean' in d:
                        line['Discrete '+metric] = d['boolean'][metric]
                    else:
                        line['Discrete '+metric] = None
            line_round = {}
            for k, v in line.items():
                if isinstance(v, float):
                    line_round[k] = round(v*100, 2)
                else:
                    line_round[k] = v
            lines.append(line_round)
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