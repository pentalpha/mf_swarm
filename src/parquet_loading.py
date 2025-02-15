import json
import polars as pl
from typing import List
from os import path, listdir

class VectorLoader:
    def __init__(self, parquet_directory: str):
        """
        Initialize the VectorLoader with the directory containing Parquet files.

        :param parquet_directory: Path to the directory containing Parquet files.
        """
        self.parquet_directory = path.expanduser(parquet_directory)
        file_list = self._get_parquet_files()
        file_names = [path.basename(x).replace('.parquet', '').replace('emb.', '')
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
                if f.endswith('.parquet') and 'emb.' in f]

    def load_vectors_by_ids(self, ids: List[str], dataset_names: List[str]):
        if len(dataset_names) == 0:
            dataset_names = sorted(self.parquets.keys())
        loaded_data = []
        for name in dataset_names:
            print('Loading', name)
            p = self.parquets[name]
            q = (
                pl.scan_parquet(p).filter(pl.col('id').is_in(ids))
            )

            new_df = q.collect()
            loaded_embs = new_df['emb'].to_numpy()
            print(new_df.head())
            loaded_data.append(loaded_embs)
            ids = new_df['id'].to_list()

        new_df_dict = {'id': ids}
        for n, d in zip(dataset_names, loaded_data):
            new_df_dict[n] = d
        new_df = pl.DataFrame(new_df_dict)

        return new_df

# Example usage
if __name__ == "__main__":
    # Initialize the VectorLoader with the directory containing Parquet files
    loader = VectorLoader(parquet_directory='~/data/dimension_db/release_1')

    # List of IDs (as strings) to load vectors for
    ids_to_load = ["C0HLM2", "C0HM02", "P83480", "P86269", "P56207"]

    # Load vectors
    new_df = loader.load_vectors_by_ids(ids_to_load, dataset_names=[])
    print(new_df.head())

    # Print the loaded vectors
    #for vec_id, vector in zip(ids_to_load, vectors):
    #    print(f"ID: {vec_id}, Vector: {vector}")