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
    
    fold_ids = sorted(set(features_df['fold_id'].to_list()))
    train_ids_basename = output_dir + '/train_ids.txt'
    test_ids_basename = output_dir + '/test_ids.txt'
    train_x_basename = output_dir + '/train_x.obj'
    test_x_basename = output_dir + '/test_x.obj'

    flows = {fold_i: {
            'train_ids': train_ids_basename.replace('.txt', f'_{fold_i}.txt'),
            'train_x': train_x_basename.replace('.obj', f'_{fold_i}.obj'),
            'test_ids': test_ids_basename.replace('.txt', f'_{fold_i}.txt'),
            'test_x': test_x_basename.replace('.obj', f'_{fold_i}.obj')
            }
        for fold_i in fold_ids}
    
    print('Separating splits', file=sys.stderr)
    ids_splitted = []
    for fold_idx in fold_ids:
        ids_for_testing = features_df.filter(pl.col('fold_id') == fold_idx)['id'].to_list()
        ids_for_training = features_df.filter(pl.col('fold_id') != fold_idx)['id'].to_list()
        ids_splitted.append((fold_idx, ids_for_training, ids_for_testing))

    print('Using splits to create train/test folds', file=sys.stderr)
    for fold_idx, ids_for_training, ids_for_testing in ids_splitted:
        train_df = features_df.filter(pl.col('id').is_in(ids_for_training))
        test_df = features_df.filter(pl.col('id').is_in(ids_for_testing))

        train_id_to_index = {a: i for i, a in enumerate(ids_for_training)}
        test_id_to_index = {a: i for i, a in enumerate(ids_for_testing)}
        #assign new indexes to col unique_idx:
        train_df = train_df.with_columns(pl.col('id').replace(train_id_to_index).alias('unique_idx'))
        test_df = test_df.with_columns(pl.col('id').replace(test_id_to_index).alias('unique_idx'))

        #sort according to unique_idx:
        train_df = train_df.sort('unique_idx')
        test_df = test_df.sort('unique_idx')

        #keep only feature_cols:
        train_df = train_df.select(feature_cols)
        test_df = test_df.select(feature_cols)

        #save
        train_x_np = []
        for col in train_df.columns:
            train_x_np.append((col, train_df[col].to_numpy()))

        test_x_np = []
        for col in test_df.columns:
            test_x_np.append((col, test_df[col].to_numpy()))

        to_save = [(flows[fold_idx]['train_x'], train_x_np), 
                (flows[fold_idx]['test_x'], test_x_np)]
        
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))

        #save ids as .txt
        train_ids = train_df['id'].to_list()
        test_ids = test_df['id'].to_list()
        
        open(flows[fold_idx]['train_ids'], 'w').write('\n'.join(train_ids))
        open(flows[fold_idx]['test_ids'], 'w').write('\n'.join(test_ids))
    '''for test_fold_i, fold_parts_dict in flows.items():
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
            dump(obj, open(p, 'wb'))'''

    folds_description = {fold_idx: {
        'train_ids': ids_for_training,
        'test_ids': ids_for_testing
        }
        for fold_idx, ids_for_training, ids_for_testing in ids_splitted
    }
    
    return output_dir, folds_description
    

def prepare_label_folds_obj(labels_df, n_folds, output_dir, folds_description: dict):
    """
    Splits labels dataframe into .obj files for each fold.
    labels_df must have columns ['id', 'fold_id', 'labels']
    """
    print(f'Preparing Label Folds ({n_folds} folds) into {output_dir}')
    if not path.exists(output_dir):
        mkdir(output_dir)
        
    fold_ids = sorted(set(labels_df['fold_id'].to_list()))
    train_y_basename = output_dir + '/train_y.obj'
    test_y_basename = output_dir + '/test_y.obj'
    train_ids_basename = output_dir + '/train_ids.txt'
    test_ids_basename = output_dir + '/test_ids.txt'

    flows = {fold_i: {
            'train_y': train_y_basename.replace('.obj', f'_{fold_i}.obj'),
            'test_y': test_y_basename.replace('.obj', f'_{fold_i}.obj'),
            'train_ids': train_ids_basename.replace('.txt', f'_{fold_i}.txt'),
            'test_ids': test_ids_basename.replace('.txt', f'_{fold_i}.txt')
            }
        for fold_i in fold_ids}

    for fold_i, obj_paths in flows.items():
        
        ids_for_training = folds_description[fold_i]['train_ids']
        ids_for_testing = folds_description[fold_i]['test_ids']

        train_y_df = labels_df.filter(pl.col('id').is_in(ids_for_training))
        test_y_df = labels_df.filter(pl.col('id').is_in(ids_for_testing))

        train_id_to_index = {a: i for i, a in enumerate(ids_for_training)}
        test_id_to_index = {a: i for i, a in enumerate(ids_for_testing)}
        #assign new indexes to col unique_idx:
        train_y_df = train_y_df.with_columns(pl.col('id').replace(train_id_to_index).alias('unique_idx'))
        test_y_df = test_y_df.with_columns(pl.col('id').replace(test_id_to_index).alias('unique_idx'))

        #sort according to unique_idx:
        train_y_df = train_y_df.sort('unique_idx')
        test_y_df = test_y_df.sort('unique_idx')

        train_y_np = train_y_df['labels'].to_numpy()
        test_y_np = test_y_df['labels'].to_numpy()

        #Save objs
        to_save = [(obj_paths['train_y'], train_y_np), 
                (obj_paths['test_y'], test_y_np)]
        
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))

        #save ids as .txt
        train_ids = train_y_df['id'].to_list()
        test_ids = test_y_df['id'].to_list()
        
        open(obj_paths['train_ids'], 'w').write('\n'.join(train_ids))
        open(obj_paths['test_ids'], 'w').write('\n'.join(test_ids))

    '''print('Separating splits for labels', file=sys.stderr)
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
            dump(obj, open(p, 'wb'))'''
    
    return output_dir
