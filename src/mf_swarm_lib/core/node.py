from glob import glob
import json
from os import path, mkdir
from typing import List
import networkx as nx
import numpy as np
import polars as pl
from tqdm import tqdm

from mf_swarm_lib.core.ensemble import BasicEnsemble
from mf_swarm_lib.core.ml.multi_input_clf import MultiInputNet

class Node():
    def __init__(self, node_dir, model_name='standard_model'):
        self.node_dir = node_dir
        self.name = path.basename(node_dir)
        results_path = node_dir+'/standard_results.json'
        params_path = node_dir+'/standard_params.json'
        self.params = json.load(open(params_path, 'r'))
        self.results = json.load(open(results_path, 'r'))
        self.model_path = node_dir+'/'+model_name
        self.validation1_results_path = node_dir + '/standard_validation.parquet'
        self.protein_labels_list = self.params['node']['go']
        self.basic_ensemble = None

    def load_model(self):
        self.basic_ensemble = BasicEnsemble.load(self.model_path)
        #self.basic_ensemble = MultiInputNet.load()
        #self.basic_ensemble = load(open(self.model_path, 'rb'))
    
    def unload_model(self):
        del self.basic_ensemble

    def predict_scores(self, feature_lists, autounload=True, weights=[1, 2, 2, 2, 3]):
        #if self.basic_ensemble is None:
        self.load_model()
        base_model_results = [m.predict(feature_lists) 
            for m in self.basic_ensemble.models]
        results_weighted = np.average(base_model_results, axis=0)
        if autounload:
            self.unload_model()
        score_lists = base_model_results + [results_weighted]
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

def all_paths_to_root(go_id_sequence, go_network, root='GO:0003674'):
    all_paths = {}
    print("Calculating all paths to root")
    for go_id in tqdm(go_id_sequence):
        all_paths[go_id] = []
        if go_id == root:
            all_paths[go_id] = [root]
        else:
            paths = list(nx.all_simple_paths(go_network, source=go_id, target=root))
            if paths:
                paths_raw = [path for path in paths if len(path) > 1]
                paths = []
                for p in paths_raw:
                    p.remove(root)  # Remove root from the path
                    p.remove(go_id)  # Remove the current GO ID
                    p2 = [x for x in p if x in go_id_sequence]  # Filter to keep only valid GO IDs
                    if len(p2) > 0:
                        paths.append(p2)
                paths.sort(key=lambda x: len(x))  # Sort paths by length
                all_paths[go_id] = paths
    return all_paths