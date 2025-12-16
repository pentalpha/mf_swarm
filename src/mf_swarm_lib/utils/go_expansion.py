import networkx as nx
import json
import numpy as np
import obonet
import sys
from tqdm import tqdm
import polars as pl

import urllib

def gos_not_to_use():
    go_check_url = "http://current.geneontology.org/ontology/subsets/gocheck_do_not_annotate.json"
    with urllib.request.urlopen(go_check_url) as resp:
        notuse_json = json.load(resp)
        #notuse_json = json.load(open(go_not_use_path, 'r'))
        nodes = notuse_json['graphs'][0]['nodes']
        ids = [nodes[i]['id']
            for i in range(len(nodes))]
        goids = ['GO:'+x.split('_')[-1] for x in ids if 'GO_' in x]
        return set(goids)
    return None

def load_go_graph(go_basic_path):
    graph = obonet.read_obo(go_basic_path)
    return graph

def expand_go_set(goid: str, go_graph: nx.MultiDiGraph, goes_to_not_use: set):
    all_gos = set()

    if goid in go_graph:
        parents = sorted(nx.descendants(go_graph, goid))
        for parent in parents:
            if not parent in goes_to_not_use:
                all_gos.add(parent)
    
        all_gos.add(goid)
    
    return sorted(all_gos)

def create_expanded_df(params_dict: dict):
    '''{'df_path': df_path, 'mode': mode, 'G': G, 
            'goids_path': go_ids_path, 'not_use': not_use, 
            'output_path': output_path}'''
    
    mode = params_dict['mode']
    df = pl.read_parquet(params_dict['df_path'])
    print(params_dict['df_path'], file=sys.stderr)
    print(df, file=sys.stderr)
    gos_to_not_use = params_dict['not_use']
    goids = open(params_dict['goids_path'], 'r').read().rstrip('\n').split('\n')
    G = params_dict['G']

    new_lines = []
    all_goids = set(goids)

    operation = lambda l: float(np.mean(l))
    if mode != 'mean':
        if mode == 'min':
            operation = min
        elif mode == 'max':
            operation = max

    for row in df.rows(named=True):
        #print(row, file=sys.stderr)
        scores = {k: v for k, v in zip(goids, row['labels'])}
        id = row['id']

        new_scores = {}
        for goid, score in scores.items():
            if score > 0.1:
                parents = expand_go_set(goid, G, gos_to_not_use)
                new_goids = [p for p in parents if not p in scores]
                for new_go in new_goids:
                    if not new_go in new_scores:
                        new_scores[new_go] = [score]
                    else:
                        new_scores[new_go].append(score)
        
        for goid, children_scores in new_scores.items():
            scores[goid] = operation(children_scores)
            all_goids.add(goid)
        new_lines.append({'id': id, 'scores_dict': scores})
    
    new_goids_sequence = sorted(all_goids)

    go_scores_matrix = []
    uniprot_ids = []
    for n in new_lines:
        g_dict = n['scores_dict']
        u = n['id']
        labels_list = [g_dict[g] if g in g_dict else 0.0 for g in new_goids_sequence]
        go_scores_matrix.append(np.array(labels_list))
        uniprot_ids.append(u)
    
    new_df = pl.DataFrame({
        'id': uniprot_ids,
        'labels': go_scores_matrix
    })
    print(new_df)

    open(params_dict['output_path'].replace('-preds.parquet', '-label_names.txt'), 'w').write(
        '\n'.join(new_goids_sequence))
    new_df.write_parquet(params_dict['output_path'])
