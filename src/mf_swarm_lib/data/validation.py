"""
Dataset validation utilities for verifying .obj files against .parquet files.

This module handles validation of both feature and label fold files, ensuring
data integrity after dataset generation.
"""

from os import path
import numpy as np
import polars as pl


def validate_feature_folds(feature_parquet_path, feature_folds_dir, n_folds=5):
    """Validate that .obj feature folds match the source .parquet file
    
    Args:
        feature_parquet_path: Path to features_traintest.parquet
        feature_folds_dir: Directory containing train_x_*.obj and test_x_*.obj files
        n_folds: Number of folds to validate (default: 5)
    
    Returns:
        bool: True if all validations passed, False otherwise
    """
    from pickle import load as pload
    
    print(f"\nValidating features: {path.basename(feature_parquet_path)}")
    df = pl.read_parquet(feature_parquet_path)
    
    all_passed = True
    for fold_i in range(n_folds):
        fold_str = str(fold_i)
        train_path = path.join(feature_folds_dir, f"train_x_{fold_str}.obj")
        test_path = path.join(feature_folds_dir, f"test_x_{fold_str}.obj")
        
        if not path.exists(train_path) or not path.exists(test_path):
            print(f"  ❌ Fold {fold_i}: Missing fold files")
            all_passed = False
            continue
        
        with open(train_path, 'rb') as f:
            train_data = pload(f)
        with open(test_path, 'rb') as f:
            test_data = pload(f)
        
        # Validate train and test splits
        for split_name, obj_data, is_test in [('train', train_data, False), ('test', test_data, True)]:
            data_dict = {col: arr for col, arr in obj_data}
            
            if is_test:
                target_df = df.filter(pl.col('fold_id') == fold_i)
            else:
                target_df = df.filter(pl.col('fold_id') != fold_i)
            
            n_rows_obj = list(data_dict.values())[0].shape[0]
            n_rows_df = target_df.height
            
            if n_rows_obj != n_rows_df:
                print(f"  ❌ Fold {fold_i} {split_name}: Row count mismatch (obj={n_rows_obj}, df={n_rows_df})")
                all_passed = False
                continue
            
            if 'id' not in data_dict:
                print(f"  ⚠ Fold {fold_i} {split_name}: No 'id' column in .obj, skipping validation")
                continue
            
            obj_ids = data_dict['id']
            df_ids = target_df['id'].to_numpy()
            
            # Create sort indices (pl.concat() reorders rows differently than filter())
            obj_sort_idx = np.argsort(obj_ids)
            df_sort_idx = np.argsort(df_ids)
            
            # Verify IDs match after sorting
            if not np.array_equal(obj_ids[obj_sort_idx], df_ids[df_sort_idx]):
                print(f"  ❌ Fold {fold_i} {split_name}: Different protein IDs in .obj vs .parquet")
                all_passed = False
                continue
            
            # Check data equality for each column (after sorting)
            for col_name, arr in data_dict.items():
                if col_name == 'id':
                    continue  # Already validated
                if col_name not in target_df.columns:
                    continue
                
                # Sort both arrays by ID order
                arr_sorted = arr[obj_sort_idx] if arr.ndim == 1 else arr[obj_sort_idx, :]
                
                # Get from dataframe and sort
                col_series = target_df[col_name]
                df_arr = col_series.to_numpy()
                df_arr_sorted = df_arr[df_sort_idx] if df_arr.ndim == 1 else df_arr[df_sort_idx, :]
                
                # Compare
                if arr_sorted.dtype.kind == 'f' or df_arr_sorted.dtype.kind == 'f':
                    if not np.allclose(arr_sorted, df_arr_sorted, equal_nan=True, rtol=1e-9, atol=1e-9):
                        print(f"  ❌ Fold {fold_i} {split_name}: Data mismatch in column {col_name}")
                        _show_float_mismatch_details(arr_sorted, df_arr_sorted)
                        all_passed = False
                        break
                else:
                    if not np.array_equal(arr_sorted, df_arr_sorted):
                        print(f"  ❌ Fold {fold_i} {split_name}: Data mismatch in column {col_name}")
                        _show_mismatch_details(arr_sorted, df_arr_sorted)
                        all_passed = False
                        break
    
    if all_passed:
        print(f"  ✓ All {n_folds} folds validated successfully")
    return all_passed


def validate_label_folds(labels_parquet_path, labels_folds_dir, cluster_name, n_folds=5):
    """Validate that .obj label folds match the source .parquet file
    
    Args:
        labels_parquet_path: Path to labels_traintest_<cluster>.parquet
        labels_folds_dir: Directory containing train_y_*.obj and test_y_*.obj files
        cluster_name: Name of the cluster (for display)
        n_folds: Number of folds to validate (default: 5)
    
    Returns:
        bool: True if all validations passed, False otherwise
    """
    from pickle import load as pload
    
    print(f"\nValidating labels: {cluster_name}")
    df = pl.read_parquet(labels_parquet_path)
    
    all_passed = True
    for fold_i in range(n_folds):
        fold_str = str(fold_i)
        train_path = path.join(labels_folds_dir, f"train_y_{fold_str}.obj")
        test_path = path.join(labels_folds_dir, f"test_y_{fold_str}.obj")
        
        if not path.exists(train_path) or not path.exists(test_path):
            print(f"  ❌ Fold {fold_i}: Missing fold files")
            all_passed = False
            continue
        
        with open(train_path, 'rb') as f:
            train_y = pload(f)
        with open(test_path, 'rb') as f:
            test_y = pload(f)
        
        # Get corresponding dataframes
        test_df = df.filter(pl.col('fold_id') == fold_i)
        train_df = df.filter(pl.col('fold_id') != fold_i)
        
        # Convert labels column to numpy
        train_df_y = np.array(train_df['labels'].to_list())
        test_df_y = np.array(test_df['labels'].to_list())
        
        # Validate shapes and data
        if train_y.shape != train_df_y.shape:
            print(f"  ❌ Fold {fold_i} train: Shape mismatch (obj={train_y.shape}, df={train_df_y.shape})")
            all_passed = False
        elif not np.array_equal(train_y, train_df_y):
            print(f"  ❌ Fold {fold_i} train: Data mismatch")
            all_passed = False
        
        if test_y.shape != test_df_y.shape:
            print(f"  ❌ Fold {fold_i} test: Shape mismatch (obj={test_y.shape}, df={test_df_y.shape})")
            all_passed = False
        elif not np.array_equal(test_y, test_df_y):
            print(f"  ❌ Fold {fold_i} test: Data mismatch")
            all_passed = False
    
    if all_passed:
        print(f"  ✓ All {n_folds} folds validated successfully")
    return all_passed


def validate_dataset_integrity(dataset, datasets_dir, n_folds=5):
    """
    Validate all generated .obj files against their source .parquet files.
    
    Args:
        dataset: Dataset object with go_clusters and dataset_name attributes
        datasets_dir: Directory containing datasets
        n_folds: Number of folds to validate (default: 5)
    
    Returns:
        bool: True if all validations passed, False otherwise
    """
    print("\n" + "="*60)
    print("DATASET INTEGRITY VALIDATION")
    print("="*60)
    
    print("="*60)
    
    # Check if we are already pointing to the dataset root (e.g. from dataset_maker.py)
    if path.exists(path.join(datasets_dir, 'params.json')):
        outputdir = datasets_dir
    else:
        outputdir = path.join(datasets_dir, dataset.dataset_name)
        
    all_validations_passed = True
    
    # Validate global feature folds
    print("\n[1/3] Validating Global Feature Folds...")
    feature_parquet = path.join(outputdir, 'features_traintest.parquet')
    feature_folds_dir = path.join(outputdir, 'features_traintest_folds5')
    
    if path.exists(feature_parquet) and path.exists(feature_folds_dir):
        passed = validate_feature_folds(feature_parquet, feature_folds_dir, n_folds)
        all_validations_passed = all_validations_passed and passed
    else:
        print(f"  ⚠ Feature files not found")
        print(f"    Looking for: {feature_parquet}")
        print(f"    Looking for: {feature_folds_dir}")
        all_validations_passed = False
    
    # Validate label folds for each cluster
    print(f"\n[2/3] Validating Label Folds ({len(dataset.go_clusters)} clusters)...")
    for cluster_name in dataset.go_clusters.keys():
        labels_parquet = path.join(outputdir, f'labels_traintest_{cluster_name}.parquet')
        labels_folds_dir = path.join(outputdir, f'labels_traintest_{cluster_name}_folds5')
        
        if path.exists(labels_parquet) and path.exists(labels_folds_dir):
            passed = validate_label_folds(labels_parquet, labels_folds_dir, cluster_name, n_folds)
            all_validations_passed = all_validations_passed and passed
        else:
            print(f"  ⚠ Skipping {cluster_name}: Files not found")
    
    # Summary
    print("\n[3/3] Validation Summary")
    print("="*60)
    if all_validations_passed:
        print("✓ ALL VALIDATIONS PASSED - Dataset integrity verified!")
    else:
        print("❌ SOME VALIDATIONS FAILED - Please review errors above")
    print("="*60 + "\n")
    
    return all_validations_passed


def _show_float_mismatch_details(arr_sorted, df_arr_sorted):
    """Helper to show detailed mismatch information for float arrays"""
    diff_mask = ~np.isclose(arr_sorted, df_arr_sorted, equal_nan=True, rtol=1e-9, atol=1e-9)
    if arr_sorted.ndim == 2:
        diff_rows, diff_cols = np.where(diff_mask)
        if len(diff_rows) > 0:
            first_diff_row = diff_rows[0]
            first_diff_col = diff_cols[0]
            print(f"     First mismatch at row {first_diff_row}, col {first_diff_col}:")
            print(f"     .obj value: {arr_sorted[first_diff_row, first_diff_col]}")
            print(f"     .parquet value: {df_arr_sorted[first_diff_row, first_diff_col]}")
            print(f"     Showing first row in .obj:  {arr_sorted[first_diff_row, :10]}")
            print(f"     Showing first row in .parquet: {df_arr_sorted[first_diff_row, :10]}")
            total_diffs = np.sum(diff_mask)
            print(f"     Total mismatches: {total_diffs} out of {arr_sorted.size} values")
    else:
        diff_idx = np.where(diff_mask)[0]
        if len(diff_idx) > 0:
            first_idx = diff_idx[0]
            print(f"     First mismatch at index {first_idx}:")
            print(f"     .obj value: {arr_sorted[first_idx]}")
            print(f"     .parquet value: {df_arr_sorted[first_idx]}")


def _show_mismatch_details(arr_sorted, df_arr_sorted):
    """Helper to show detailed mismatch information for non-float arrays"""
    diff_mask = arr_sorted != df_arr_sorted
    if arr_sorted.ndim == 2:
        diff_rows, diff_cols = np.where(diff_mask)
        if len(diff_rows) > 0:
            first_diff_row = diff_rows[0]
            first_diff_col = diff_cols[0]
            print(f"     First mismatch at row {first_diff_row}, col {first_diff_col}:")
            print(f"     .obj value: {arr_sorted[first_diff_row, first_diff_col]}")
            print(f"     .parquet value: {df_arr_sorted[first_diff_row, first_diff_col]}")
            total_diffs = np.sum(diff_mask)
            print(f"     Total mismatches: {total_diffs} out of {arr_sorted.size} values")
