
import sys
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.data.dataset import Dataset, dataset_types

if __name__ == "__main__":
    dimension_db_releases_dir = sys.argv[1]
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #30
    min_proteins_per_mf    = int(sys.argv[4])
    val_perc    = float(sys.argv[5])
    dataset_type           = sys.argv[6]

    assert dataset_type in dataset_types

    dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, 
        new_downloads=False)
    dataset = Dataset(dimension_db=dimension_db, min_proteins_per_mf=min_proteins_per_mf, 
        dataset_type=dataset_type, val_perc=val_perc)
    if dataset.new_dataset:
        dataset.save(datasets_dir)
        
        # Validate dataset integrity after creation
        print("\n" + "="*70)
        print("Dataset saved successfully. Running integrity validation...")
        print("="*70)
        validation_passed = dataset.validate_dataset_integrity(datasets_dir)
        
        if not validation_passed:
            print("\n⚠️  WARNING: Some validation checks failed!")
            print("Please review the errors above before using this dataset.")
            sys.exit(1)
        else:
            print("\n✅ Dataset creation completed successfully with full integrity verification!")