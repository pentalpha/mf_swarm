from os import path
from mf_swarm_lib.data.parquet_loading import VectorLoader, load_columns_from_parquet

# Example usage
if __name__ == "__main__":
    # Initialize the VectorLoader with the directory containing Parquet files
    loader = VectorLoader(parquet_directory=path.expanduser("~/data/dimension_db/release_1"))

    all_ids = load_columns_from_parquet(path.expanduser("~/data/dimension_db/release_1/emb.prottrans.parquet"),
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