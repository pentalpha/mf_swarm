import sys
from os import path, mkdir
from pickle import dump
import numpy as np
import polars as pl

def validate_folds(traintest_df: pl.DataFrame, n_folds: int, cluster_go_proper_order):
    if n_folds == 1:
        return traintest_df, set()

    parts = [traintest_df.filter(pl.col('fold_id') == i) for i in range(n_folds)]
    
    constant_labels = set()
    for part in parts:
        labels_np = part['labels'].to_numpy()
        nrows = labels_np.shape[0]
        if nrows == 0: 
             continue
        labels_sum = np.sum(labels_np, axis=0)
        non_constant = [s > 0 and s < nrows for s in labels_sum]
        for go_id, is_non_constant in zip(cluster_go_proper_order, non_constant):
            if not is_non_constant:
                constant_labels.add(go_id)
    
    if len(constant_labels) > 0:
        print('Constant labels found in pre-defined split:', len(constant_labels))
        # print('Constant labels:', constant_labels)

    return traintest_df, constant_labels


def prepare_global_feature_folds(features_df, n_folds, output_dir, feature_cols):
    print(f'Preparing Global Feature Folds ({n_folds} folds) into {output_dir}')
    if not path.exists(output_dir):
        mkdir(output_dir)
    
    fold_ids = [str(i) for i in range(n_folds)]
    train_x_basename = output_dir + '/train_x.obj'
    test_x_basename = output_dir + '/test_x.obj'

    flows = {fold_i: {
            'train_x': train_x_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_x': test_x_basename.replace('.obj', '_'+fold_i+'.obj')
            }
        for fold_i in fold_ids}
    
    print('Separating splits', file=sys.stderr)
    traintest_parts = []
    for fold_idx in range(n_folds):
        part = features_df.filter(pl.col('fold_id') == fold_idx)
        traintest_parts.append(part)

    print('Using splits to create train/test folds', file=sys.stderr)
    for test_fold_i, fold_parts_dict in flows.items():
        train_parts = [part 
            for i, part in enumerate(traintest_parts) 
            if i != int(test_fold_i)]
        test = traintest_parts[int(test_fold_i)]
        train = pl.concat(train_parts)
    
        train_x = train.select(feature_cols)
        test_x = test.select(feature_cols)

        train_x_np = []
        for col in train_x.columns:
            train_x_np.append((col, train_x[col].to_numpy()))

        test_x_np = []
        for col in test_x.columns:
            test_x_np.append((col, test_x[col].to_numpy()))
        
        to_save = [(fold_parts_dict['train_x'], train_x_np), 
                (fold_parts_dict['test_x'], test_x_np)]
        
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))
    
    return output_dir

def prepare_label_folds_obj(labels_df, n_folds, output_dir):
    """
    Splits labels dataframe into .obj files for each fold.
    labels_df must have columns ['id', 'fold_id', 'labels']
    """
    print(f'Preparing Label Folds ({n_folds} folds) into {output_dir}')
    if not path.exists(output_dir):
        mkdir(output_dir)
        
    fold_ids = [str(i) for i in range(n_folds)]
    train_y_basename = output_dir + '/train_y.obj'
    test_y_basename = output_dir + '/test_y.obj'

    flows = {fold_i: {
            'train_y': train_y_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_y': test_y_basename.replace('.obj', '_'+fold_i+'.obj')
            }
        for fold_i in fold_ids}

    print('Separating splits for labels', file=sys.stderr)
    # Using the fold_id column in labels_df
    traintest_parts = []
    for fold_idx in range(n_folds):
        part = labels_df.filter(pl.col('fold_id') == fold_idx)
        traintest_parts.append(part)

    print('Using splits to create train/test label folds', file=sys.stderr)
    for test_fold_i, fold_parts_dict in flows.items():
        train_parts = [part 
            for i, part in enumerate(traintest_parts) 
            if i != int(test_fold_i)]
        test = traintest_parts[int(test_fold_i)]
        train = pl.concat(train_parts)
    
        train_y_np = train['labels'].to_numpy()
        test_y_np = test['labels'].to_numpy()
        
        to_save = [(fold_parts_dict['train_y'], train_y_np),
                (fold_parts_dict['test_y'], test_y_np)]
        
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))
    
    return output_dir
