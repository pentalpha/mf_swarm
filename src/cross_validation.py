from os import path
import sys
import numpy as np
from pickle import load, dump

from util_base import concat_lists, run_command
from parquet_loading import load_columns_from_parquet
from node_factory import makeMultiClassifierModel, split_into_n_parts, split_train_test_polars
import polars as pl
class BasicEnsemble():
    def __init__(self, model_list) -> None:
        self.models = model_list
    
    def predict(self, x, verbose=0):
        results = [m.predict(x) for m in self.models]
        results_mean = np.mean(results, axis=0)
        return results_mean
    
def split_train_test_n_folds(traintest_path, features, max_proteins=60000):
    #Make sure the dataset is prepared as train_x, train_y, test_x and test_y binary files
    #If not prepared, converts the parquet file to separate binary files and saves them
    fold_ids = load_columns_from_parquet(traintest_path, ['fold_id'])
    n_folds = fold_ids['fold_id'].max() + 1
    base_dir = traintest_path.replace('.parquet', '_folds'+str(n_folds)+"_features-"+'-'.join(features))
    
    fold_ids = [str(i) for i in range(n_folds)]

    train_x_basename = base_dir + '/train_x.obj'
    train_y_basename = base_dir + '/train_y.obj'
    test_x_basename = base_dir + '/test_x.obj'
    test_y_basename = base_dir + '/test_y.obj'

    folds = {fold_i: {
            'train_x': train_x_basename.replace('.obj', '_'+fold_i+'.obj'),
            'train_y': train_y_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_x': test_x_basename.replace('.obj', '_'+fold_i+'.obj'),
            'test_y': test_y_basename.replace('.obj', '_'+fold_i+'.obj')
            }
        for fold_i in fold_ids}

    all_paths = concat_lists([list(f.values()) for f in folds.values()])

    if all([path.exists(p) for p in all_paths]):
        return True
    else:
        print('Preparing', base_dir)
        run_command(['mkdir -p', base_dir])
        cols_to_use = ['id', 'fold_id'] + features + ['labels']
        traintest = load_columns_from_parquet(traintest_path, cols_to_use)
        '''if len(traintest) > max_proteins:
            print(traintest_path, 'is too large, sampling down')
            traintest = traintest.sample(fraction=(max_proteins/len(traintest)), 
                shuffle=True, seed=1337+int(fold_i))'''

        print('Separating splits', file=sys.stderr)
        # Cria os folds usando a coluna fold_id já existente
        traintest_parts = []
        for fold_idx in range(n_folds):
            part = traintest.filter(pl.col('fold_id') == fold_idx)
            traintest_parts.append(part)

        print('Using splits to create train/test folds', file=sys.stderr)
        for test_fold_i, fold_parts_dict in folds.items():
            train_parts = [part 
                for i, part in enumerate(traintest_parts) 
                if i != int(test_fold_i)]
            test = traintest_parts[int(test_fold_i)]
            train = pl.concat(train_parts)
        
            feature_columns = [c for c in train.columns if not c in ['labels', 'fold_id', 'id']]
            train_ids = train['id'].to_list()
            train_x = train.select(feature_columns)
            train_y = train.select('labels')
            test_ids = test['id'].to_list()
            test_x = test.select(feature_columns)
            test_y = test.select('labels')

            train_x_np = []
            for col in train_x.columns:
                train_x_np.append((col, train_x[col].to_numpy()))
            train_y_np = train_y['labels'].to_numpy()

            test_x_np = []
            for col in test_x.columns:
                test_x_np.append((col, test_x[col].to_numpy()))
            test_y_np = test_y['labels'].to_numpy()
            
            to_save = [(fold_parts_dict['train_x'], train_x_np), 
                    (fold_parts_dict['train_y'], train_y_np),
                    (fold_parts_dict['test_x'], test_x_np),
                    (fold_parts_dict['test_y'], test_y_np)]
            if not path.exists('tmp'):
                run_command(['mkdir', 'tmp'])
            for p, obj in to_save:
                dump(obj, open(p, 'wb'))

            open(base_dir+'/train_ids_'+test_fold_i+'.txt', 'w').write('\n'.join(train_ids))
            open(base_dir+'/test_ids_'+test_fold_i+'.txt', 'w').write('\n'.join(test_ids))
        
        return True
    
def train_crossval_node(params: dict, features: list, n_folds: int, max_proteins=60000):
    node = params['node']
    params_dict = params['params_dict']

    traintest_path = node['traintest_path']
    base_dir = traintest_path.replace('.parquet', 
        '_folds'+str(n_folds)+"_features-"+'-'.join(features))
    #print('loading', base_dir)
    
    models = []
    stats_dicts = []
    for fold_i in range(n_folds):
        fold_i_str = str(fold_i)

        train_x_name = base_dir + '/train_x_'+fold_i_str+'.obj'
        train_y_name = base_dir + '/train_y_'+fold_i_str+'.obj'
        test_x_name = base_dir + '/test_x_'+fold_i_str+'.obj'
        test_y_name = base_dir + '/test_y_'+fold_i_str+'.obj'

        train_x = load(open(train_x_name, 'rb'))
        train_y = load(open(train_y_name, 'rb'))
        test_x = load(open(test_x_name, 'rb'))
        test_y = load(open(test_y_name, 'rb'))

        annot_model, stats = makeMultiClassifierModel(train_x, train_y, test_x, test_y, 
            params_dict)
        models.append(annot_model)
        stats_dicts.append(stats)
    
    stats = {}
    for k in stats_dicts[0].keys():
        stats[k] = np.mean([d[k] for d in stats_dicts])

    #print(params['cluster_name'], stats)
    annot_model = BasicEnsemble(models)

    return annot_model, stats

class CrossValRunner():
    def __init__(self, param_translator, node, features, n_folds):
        self.new_param_translator = param_translator
        self.node = node
        self.features = features
        self.n_folds = n_folds
        
    def objective_func(self, solution):
        #print('objective_func', file=sys.stderr)
        new_params_dict = self.new_param_translator.decode(solution)
        self.node['params_dict'] = new_params_dict
        #print('getting roc_auc s', file=sys.stderr)
        model_obj, metrics = train_crossval_node(self.node, self.features, n_folds=self.n_folds)
        print(metrics)
        #print(roc_aucs, file=sys.stderr)
        #print('objective_func finish', file=sys.stderr)
        return metrics