import sys
import json
import os
from os import path
from pickle import dump

print("New thread", file=sys.stderr)
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from cross_validation import split_train_test_n_folds, train_crossval_node, validate_cv_model_noretrain

if __name__ == '__main__':
    print(sys.argv)

    params_json_path = sys.argv[1]
    results_json_path = sys.argv[2]
    local_dir = path.dirname(results_json_path)

    params_dict = json.load(open(params_json_path, 'r'))
    n_folds = params_dict['n_folds']
    n_jobs = params_dict['n_jobs']
    node_name = params_dict['node_name']
    node = params_dict['node']
    features = params_dict['features']
    metaparameters = params_dict['params_dict']
    print(features)
    print(node_name)
    
    split_train_test_n_folds(node['traintest_path'], features)
    
    #print('getting roc_auc s', file=sys.stderr)
    model_obj = train_crossval_node(params_dict, features, n_folds=n_folds)
    val_path = node['val_path']
    go_labels = node['go']
    print(model_obj.stats)
    results, validation_solved_df = validate_cv_model_noretrain(model_obj, val_path, go_labels, features)
    
    validation_solved_df.write_parquet(local_dir+'/standard_validation.parquet')
    json.dump(results, open(results_json_path, 'w'), indent=4)
    dump(model_obj, open(local_dir+'/standard_model.obj', 'wb'))