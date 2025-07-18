import polars as pl
import numpy as np
import sys
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    prot_dimension_db_release_path = sys.argv[1]
    min_protein_annots = sys.argv[2]
    val_part = sys.argv[3]
    deepgose_dfs_dir = sys.argv[4]
    output_dir = sys.argv[5]

    validation_split_dir = prot_dimension_db_release_path + f'/validation_splits/min_prot{min_protein_annots}_val{val_part}'
    val_ids = validation_split_dir+'/validation_ids.txt'

    validation_proteins = open(val_ids, 'r').read().split('\n')

    deepgose_df_paths = glob(deepgose_dfs_dir+'/*_mf.tsv.gz')
    print('Scanning for all GO ids')
    all_gos = set()
    for df_path in tqdm(deepgose_df_paths):
        df = pl.read_csv(df_path, separator='\t', new_columns = ['id', 'go', 'score'], has_header=False)
        all_gos.update(df['go'].to_list())
    go_list = sorted(all_gos)
    go_pos = {go: index for index, go in enumerate(go_list)}
    print('Found', len(go_list), 'go ids')
    open(output_dir+'/deepgose_validation-label_names.txt', 'w').write('\n'.join(go_list))

    print('Loading annotations')
    dfs = []
    for df_path in tqdm(deepgose_df_paths):
        df = pl.read_csv(df_path, separator='\t', new_columns = ['id', 'go', 'score'], has_header=False)
        df_filtered = df.filter(pl.col('id').is_in(validation_proteins))
        ids = df_filtered['id']
        gos = df_filtered['go']
        scores = df_filtered['score']

        protein_scores = {}

        for uniprot, go, score in zip(ids, gos, scores):
            if uniprot not in protein_scores:
                protein_scores[uniprot] = {}
            protein_scores[uniprot][go] = score

        ids_list = []
        protein_labels_list = []
        for uniprot_id, scores in protein_scores.items():
            ids_list.append(uniprot_id)
            scores_vec = [scores[go] if go in scores else 0.0 for go in go_list]
            protein_labels_list.append(np.array(scores_vec))
        print('Found', len(ids_list), 'proteins')
        results_df = pl.DataFrame({
            'id': ids_list,
            'labels': protein_labels_list
        })
        dfs.append(results_df)

    print('Creating dataframe')
    final_results_df = pl.concat(dfs)
    final_results_df.write_parquet(output_dir + '/deepgose_validation-preds.parquet')

    
