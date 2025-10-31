import polars as pl
import numpy as np
import sys
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    prot_dimension_db_release_path = sys.argv[1]
    min_protein_annots = sys.argv[2]
    val_part = sys.argv[3]
    tale_res_dir = sys.argv[4]
    output_dir = sys.argv[5]

    validation_split_dir = prot_dimension_db_release_path + f'/validation_splits/min_prot{min_protein_annots}_val{val_part}'
    val_ids = validation_split_dir+'/validation_ids.txt'

    validation_proteins = open(val_ids, 'r').read().split('\n')
    pattern = tale_res_dir+'/*/*.fasta_results'
    print(pattern)
    taledf_paths = glob(pattern)
    print('Scanning for all GO ids')
    all_gos = set()
    for df_path in tqdm(taledf_paths):
        for rawline in open(df_path, 'r'):
            cells = [c for c in rawline.replace('\t', ' ').rstrip('\n').split() if len(c) != '']
            goid = cells[1].replace("'", '').replace('(', '').replace(')', '').replace(',', '')
            all_gos.add(goid)
    go_list = sorted(all_gos)
    go_pos = {go: index for index, go in enumerate(go_list)}
    print('Found', len(go_list), 'go ids')
    open(output_dir+'/tale_validation-label_names.txt', 'w').write('\n'.join(go_list))

    print('Loading annotations')
    dfs = []
    ids_processed = set()
    for df_path in tqdm(taledf_paths):
        ids = []
        gos = []
        scores = []

        for rawline in open(df_path, 'r'):
            cells = [c for c in rawline.replace('\t', ' ').rstrip('\n').split() if len(c) != '']
            if len(cells) >= 4:
                uniprot = cells[0]
                if not uniprot in ids_processed:
                    goid = cells[1].replace("'", '').replace('(', '').replace(')', '').replace(',', '')
                    score = float(cells[-1])
                    ids.append(uniprot)
                    gos.append(goid)
                    scores.append(score)
            else:
                print('Wrong number of cells:')
                print(cells)
                quit(1)

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
    final_results_df.write_parquet(output_dir + '/tale_validation-preds.parquet')

    
