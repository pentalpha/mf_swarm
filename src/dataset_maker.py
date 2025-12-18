
import sys
import argparse
from mf_swarm_lib.data.dimension_db import DimensionDB
from mf_swarm_lib.data.dataset import Dataset, dataset_types

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MF Swarm Dataset Maker CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-d', '--dimension-db-releases-dir',
        required=True,
        help="Path to the dimension database releases directory"
    )

    parser.add_argument(
        '-r', '--dimension-db-release-n',
        required=True,
        help="Dimension DB release number/identifier"
    )

    parser.add_argument(
        '-o', '--dataset-dir',
        required=True,
        help="Output directory for the generated dataset"
    )

    parser.add_argument(
        '-m', '--min-proteins-per-mf',
        type=int,
        required=True,
        help="Minimum number of proteins per Molecular Function (MF)"
    )

    parser.add_argument(
        '-v', '--val-perc',
        type=float,
        required=True,
        help="Validation set percentage (0.0 to 1.0)"
    )

    parser.add_argument(
        '-t', '--dataset-type',
        required=True,
        choices=sorted(list(dataset_types)),
        help=f"Type of dataset to create. Options: {', '.join(sorted(list(dataset_types)))}"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    dimension_db_releases_dir = args.dimension_db_releases_dir
    dimension_db_release_n = args.dimension_db_release_n
    dataset_dir = args.dataset_dir
    min_proteins_per_mf = args.min_proteins_per_mf
    val_perc = args.val_perc
    dataset_type = args.dataset_type

    print(f"Initializing Dataset Creation...")
    print(f"  Type: {dataset_type}")
    print(f"  DB Release: {dimension_db_release_n}")
    print(f"  Min Proteins/MF: {min_proteins_per_mf}")
    print(f"  Validation %: {val_perc}")
    print(f"  Output Dir: {dataset_dir}")

    # assert dataset_type in dataset_types # Handled by argparse choices

    dimension_db = DimensionDB(dimension_db_releases_dir, dimension_db_release_n, 
        new_downloads=False)
    dataset = Dataset(dimension_db=dimension_db, min_proteins_per_mf=min_proteins_per_mf, 
        dataset_type=dataset_type, val_perc=val_perc)
    
    if dataset.new_dataset:
        dataset.save_to_dir(dataset_dir)
        
        # Validate dataset integrity after creation
        print("\n" + "="*70)
        print("Dataset saved successfully. Running integrity validation...")
        print("="*70)
        validation_passed = dataset.validate_dataset_integrity(dataset_dir)
        
        if not validation_passed:
            print("\n⚠️  WARNING: Some validation checks failed!")
            print("Please review the errors above before using this dataset.")
            sys.exit(1)
        else:
            print("\n✅ Dataset creation completed successfully with full integrity verification!")
    else:
        print("\nDataset already exists or was loaded from cache. No new files created.")