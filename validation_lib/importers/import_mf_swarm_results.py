import json
import os
import polars as pl
import numpy as np
import sys
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    mf_swarm_trained_dir = sys.argv[1]
    output_dir = sys.argv[2]

    node_dirs = glob(mf_swarm_trained_dir+'/Level*')
    print('Scanning for all GO ids')
    all_gos = set()
    go_lists = []
    for node_dir in node_dirs:
        #optimized_val_path = node_dir + '/optimized_validation.parquet'
        #val_path = node_dir + '/standard_validation.parquet'
        #val_path = optimized_val_path if os.path.exists(optimized_val_path) else val_path
        exp_params_json_path = node_dir + '/exp_params.json'
        std_params_json_path = node_dir + '/standard_params.json'
        std_params_json_path = exp_params_json_path if os.path.exists(exp_params_json_path) else std_params_json_path
        exp_params = json.load(open(std_params_json_path, 'r'))
        protein_labels_list = exp_params['node']['go']
        go_lists.append(protein_labels_list)
        all_gos.update(protein_labels_list)
    print(len(all_gos))

    final_go_seq = sorted(all_gos)
    print(final_go_seq)
    open(output_dir+'/mfswarm_validation-label_names.txt', 'w').write('\n'.join(final_go_seq))
    protein_scores = {}
    for node_dir, local_go_list in zip(node_dirs, go_lists):
        print('loading', node_dir)
        optimized_val_path = node_dir + '/optimized_validation.parquet'
        val_path = node_dir + '/standard_validation.parquet'
        val_path = optimized_val_path if os.path.exists(optimized_val_path) else val_path
        val_df = pl.read_parquet(val_path)
        ids = val_df['id'].to_list()
        score_lists = val_df['y_pred'].to_list()
        for uniprot, scores in zip(ids, score_lists):
            if not uniprot in protein_scores:
                protein_scores[uniprot] = {}
            
            for score, local_go in zip(scores, local_go_list):
                protein_scores[uniprot][local_go] = score
    
    print('Create final dataframe')
    ids = []
    all_scores = []
    for uniprot, scores_dict in protein_scores.items():
        scores_vec = [scores_dict[go] if go in scores_dict else 0.0 for go in final_go_seq]
        ids.append(uniprot)
        all_scores.append(scores_vec)
    
    df = pl.DataFrame(
        {
            'id': ids,
            'labels': all_scores
        }
    )
    df.write_parquet(output_dir + '/mfswarm_validation-preds.parquet')