import polars as pl
import numpy as np
import sys
from glob import glob
from tqdm import tqdm

def run(prot_dimension_db_release_path, min_protein_annots, val_part, interpro_res, output_dir):
    validation_split_dir = prot_dimension_db_release_path + f'/validation_splits/min_prot{min_protein_annots}_val{val_part}'
    val_ids = validation_split_dir+'/validation_ids.txt'

    validation_proteins = open(val_ids, 'r').read().split('\n')
    print('Scanning for all GO ids')
    all_gos = set()
    for rawline in open(interpro_res, 'r'):
        cells = [c for c in rawline.rstrip('\n').split('\t')]
        goids = cells[1].split(';')
        all_gos.update(goids)
    go_list = sorted(all_gos)
    go_pos = {go: index for index, go in enumerate(go_list)}
    print('Found', len(go_list), 'go ids')
    open(output_dir+'/interpro_validation-label_names.txt', 'w').write('\n'.join(go_list))

    print('Loading annotations')
    dfs = []
    ids_processed = set()
    
    ids = []
    gos = []
    scores = []

    for rawline in open(interpro_res, 'r'):
        cells = [c for c in rawline.rstrip('\n').split('\t')]
        if len(cells) == 2:
            uniprot = cells[0]
            if not uniprot in ids_processed:
                goids = cells[1].split(';')
                for goid in goids:
                    ids.append(uniprot)
                    gos.append(goid)
                    scores.append(1.0)
        else:
            print('Wrong number of cells:')
            print(cells)

    protein_scores = {}

    for uniprot, go, score in zip(ids, gos, scores):
        if uniprot not in protein_scores:
            protein_scores[uniprot] = {}
        protein_scores[uniprot][go] = score
    print(len(protein_scores))

    ids_list = []
    protein_labels_list = []
    for uniprot_id, scores in protein_scores.items():
        ids_list.append(uniprot_id)
        ids_processed.add(uniprot_id)
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
    print(final_results_df)
    print(final_results_df.shape)
    final_results_df.write_parquet(output_dir + '/interpro_validation-preds.parquet')
