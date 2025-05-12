import json
import numpy as np
import polars as pl
from typing import List
from os import path, listdir

from tqdm import tqdm

class VectorLoader:
    def __init__(self, parquet_directory: str):
        """
        Initialize the VectorLoader with the directory containing Parquet files.

        :param parquet_directory: Path to the directory containing Parquet files.
        """
        self.parquet_directory = path.expanduser(parquet_directory)
        file_list = self._get_parquet_files()
        file_names = [path.basename(x).replace('.parquet', '').replace('emb.', '').replace('onehot.', '')
                      for x in file_list]
        self.parquets = {n: f for n, f in zip(file_names, file_list)}
        print('Parquet datasets found:')
        print(json.dumps(self.parquets, indent=2))

    def _get_parquet_files(self) -> List[str]:
        """
        Get a list of Parquet files in the specified directory.

        :return: List of file paths.
        """
        return [path.join(self.parquet_directory, f) 
                for f in listdir(self.parquet_directory) 
                if f.endswith('.parquet') and ('emb.' in f or 'onehot.' in f)]

    def load_vectors_by_ids(self, ids_to_load: List[str], dataset_names: List[str], 
            remove_na = False):
        if len(dataset_names) == 0:
            dataset_names = sorted(self.parquets.keys())
        loaded_data = []
        #ids = None
        #ids_str = ""
        print('Scanning features')
        if any([not (name in self.parquets) for name in dataset_names]):
            print('Missing features:', [name for name in dataset_names if name not in self.parquets])
            print('Available features:', self.parquets.keys())
            raise Exception("Missing features")
        
        id_to_index = {id: i for i, id in enumerate(ids_to_load)}

        for name in tqdm(dataset_names):
            print('Scanning', name)
            p = self.parquets[name]
            q = (
                pl.scan_parquet(p).filter(pl.col('id').is_in(ids_to_load))
            )
            print('Collecting', name)
            new_df = q.collect()
            print('Collected', name)
            loaded_embs = new_df['emb'].to_list()
            emb_width = len(loaded_embs[0])
            ids = new_df['id'].to_list()
            embs_sorted = [np.nan * emb_width] * len(ids_to_load)
            print('Sorting', name)
            for uniprot_id, emb in zip(ids, loaded_embs):
                embs_sorted[id_to_index[uniprot_id]] = emb
            print('Sorted', name)
            loaded_data.append(embs_sorted)
            #print(new_df)
            '''loaded_data.append(loaded_embs)

            if ids == None:
                ids = new_df['id'].to_list()
                ids_str = ' '.join(ids)
            else:
                new_ids = new_df['id'].to_list()
                new_ids_str = ' '.join(new_ids)
                if new_ids_str == ids_str:
                    ids = new_ids
                else:
                    raise Exception("Difference in IDs between " 
                        + name + " and " + dataset_names[0])'''
        
        if remove_na:
            na_indexes = set()
            for index, protein_id in enumerate(ids_to_load):
                for vec in loaded_data:
                    emb = vec[index]
                    if np.isnan(emb).any():
                        na_indexes.add(index)
                        break
            print(len(na_indexes), 'rows with nan values in total of', len(ids_to_load), 'rows')
            
            if len(na_indexes) > 0:
                print('Found', len(na_indexes), 'nan values')
                mask = [i for i in range(len(ids_to_load)) if not i in na_indexes]
                ids_to_load = [ids_to_load[i] for i in mask]
                for dataset_n, dataset in enumerate(loaded_data):
                    loaded_data[dataset_n] = [dataset[i] for i in mask]

        new_df_dict = {'id': ids_to_load}
        print('Converting to numpy arrays')
        for n, d in zip(dataset_names, loaded_data):
            new_df_dict[n] = np.asarray([np.array(e) for e in d])
        print('Converted to numpy arrays')
        new_df = pl.DataFrame(new_df_dict)

        return new_df

def load_columns_from_parquet(file_path: str, column_names: List[str]) -> pl.DataFrame:
    """
    Load a list of columns from a Parquet file using Polars' lazy API.

    :param file_path: Path to the Parquet file.
    :param column_names: List of column names to load.
    :return: A Polars DataFrame containing only the specified columns.
    """
    # Use the lazy API to scan the Parquet file
    lazy_frame = pl.scan_parquet(file_path)
    
    # Select the specified columns
    lazy_frame = lazy_frame.select([pl.col(col) for col in column_names])
    
    # Collect the result (execute the query)
    result = lazy_frame.collect()
    
    return result

# Example usage
if __name__ == "__main__":
    # Initialize the VectorLoader with the directory containing Parquet files
    loader = VectorLoader(parquet_directory='~/data/dimension_db/release_1')

    all_ids = load_columns_from_parquet('~/data/dimension_db/release_1/emb.prottrans.parquet',
        ['id'])['id'].to_list()
    print(all_ids[-2], all_ids[-1])

    # List of IDs (as strings) to load vectors for
    ids_to_load = ["C0HLM2", "C0HM02", "P83480", "P86269", "P56207", "Q8K2Q5"
        "Q8WZ42", "A2ASS6"]

    # Load vectors
    new_df = loader.load_vectors_by_ids(ids_to_load, dataset_names=[], remove_na=True)
    print(new_df.head(8))

    # Print the loaded vectors
    #for vec_id, vector in zip(ids_to_load, vectors):
    #    print(f"ID: {vec_id}, Vector: {vector}")