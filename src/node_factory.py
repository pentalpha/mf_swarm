
import json
import os
from os import path
import sys
import polars as pl
import numpy as np
from pickle import load, dump

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Concatenate, Input
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import Model
from keras import backend as K
print(keras.__version__)
from sklearn import metrics

from metaheuristics import ProblemTranslator, param_bounds, RandomSearchMetaheuristic
from parquet_loading import load_columns_from_parquet
from util_base import run_command, plm_sizes


def split_train_test(ids, X, Y):

    print('Train/test is', len(ids))
    total_with_val = len(ids) / (1.0 - config['validation_perc'])
    print('Total proteins was', total_with_val)
    test_n = total_with_val*config['testing_perc']
    print('Test total should be', test_n)
    testing_perc_local = test_n / len(ids)
    print('So the local testing perc. is', testing_perc_local)
    print('Splitting train and test')
    protein_indexes = np.asarray([np.array([i]) for i in range(len(ids))])
    train_ids, train_y, test_ids, test_y = iterative_train_test_split(
        protein_indexes, 
        Y, 
        test_size = testing_perc_local)
    
    train_feature_indices = [i_vec[0] for i_vec in train_ids]
    test_feature_indices = [i_vec[0] for i_vec in test_ids]
    train_x = []
    test_x = []
    for name, feature_vec in X:
        sub_vec_train = get_items_at_indexes(feature_vec, train_feature_indices)
        sub_vec_test = get_items_at_indexes(feature_vec, test_feature_indices)
        train_x.append((name, sub_vec_train))
        test_x.append((name, sub_vec_test))

    return train_ids, train_x, train_y, test_ids, test_x, test_y

def split_train_test_polars(traintest: pl.DataFrame, perc):

    #print('Train/test is', len(traintest))
    df = traintest.sample(fraction=1, shuffle=True, seed=1337)
    test_size = int(len(traintest)*perc)
    test, train = df.head(test_size), df.tail(-test_size)
    #print('test is', len(test))
    #print('train is', len(train))
    
    feature_columns = [c for c in df.columns if c != 'labels' and c != 'id']
    train_ids = train['id'].to_list()
    train_x = train.select(feature_columns)
    train_y = train.select('labels')
    test_ids = test['id'].to_list()
    test_x = test.select(feature_columns)
    test_y = test.select('labels')

    return train_ids, train_x, train_y, test_ids, test_x, test_y

def x_to_np(x):
    #print('Converting features to np')
    for i in range(len(x)):
        feature_name, feature_vec = x[i]
        x[i] = (feature_name, np.asarray([np.array(vec) for vec in feature_vec]))
        #print(feature_name, x[i][1].shape)
    return x

def makeMultiClassifierModel(train_x, train_y, test_x, test_y, params_dict):
    
    #print('go labels', train_y.shape)
    #print('go labels', test_y.shape)

    #print('Defining network')

    keras_inputs = []
    keras_input_networks = []

    for feature_name, feature_vec in train_x:
        feature_params = params_dict[feature_name]
        #print('Trying', feature_vec.shape, feature_name, feature_params)
        start_dim = feature_params['l1_dim']
        end_dim = feature_params['l2_dim']
        leakyrelu_1_alpha = feature_params['leakyrelu_1_alpha']
        dropout_rate = feature_params['dropout_rate']
        #print('1')
        input_start = Input(shape=(feature_vec.shape[1],))

        #print('2')
        input_network = Dense(start_dim, name=feature_name+'_dense_1')(input_start)
        #print('3')
        input_network = BatchNormalization(name=feature_name+'_batchnorm_1')(input_network)
        #print('4')
        input_network = LeakyReLU(negative_slope=leakyrelu_1_alpha, name=feature_name+'_leakyrelu_1')(input_network)
        
        #print('4.5')
        input_network = Dropout(dropout_rate, name=feature_name+'_dropout_1')(input_network)
        #print('5')
        input_network = Dense(end_dim, name=feature_name+'_dense_2')(input_network)
        #print('6')

        keras_inputs.append(input_start)
        #print('7')
        keras_input_networks.append(input_network)
        #print('8')
    
    final_params = params_dict['final']
    #print('9')
    final_dim = final_params['final_dim']
    dropout_rate = final_params['final_dim']
    patience = final_params['patience']
    epochs = final_params['epochs']
    learning_rate = final_params['learning_rate']
    batch_size = final_params['batch_size']

    #print("Concatenate the networks")
    combined = Concatenate()(keras_input_networks)
    #combined = LeakyReLU(alpha=0.1, name='combined_leakyrelu_1')(combined)
    #print("BatchNormalization")
    combined = BatchNormalization(name = 'combined_batchnorm_1')(combined)
    combined = Dense(final_dim, name='combined_dense_1', activation='relu')(combined)
    #print("Dense")
    output_1 = Dense(train_y.shape[1], activation='sigmoid', name='final_output_123')(combined)

    # Create the model
    #print('Creating Model')
    model = Model(inputs=keras_inputs,
        outputs=output_1)

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    
    #lr scheduling
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 10 == 0:
            lr = lr * 0.5
        return lr
    lr_callback = LearningRateScheduler(lr_schedule, verbose=0)
    es = EarlyStopping(monitor='val_loss', patience=patience)

    #print("Compiling")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
        loss = 'binary_crossentropy',
        metrics=['binary_accuracy', keras.metrics.AUC()])

    #print('Fiting')
    x_test_vec = [x for name, x in test_x]
    x_list = [x for name, x in train_x]
    #print([x.shape for x in x_list], train_y.shape)
    #print([x.shape for x in x_test_vec], test_y.shape)
    history = model.fit(x_list, 
        train_y,
        validation_data=(x_test_vec, test_y),
        epochs=epochs, batch_size=batch_size,
        callbacks=[lr_callback, es],
        verbose=1)
    
    #print('Testing')
    y_pred = model.predict(x_test_vec, verbose=0)
    roc_auc_score = metrics.roc_auc_score(test_y, y_pred)
    acc = np.mean(keras.metrics.binary_accuracy(test_y, y_pred).numpy())

    return model, {'ROC AUC': float(roc_auc_score), 'Accuracy': float(acc), 
        'Proteins': len(train_y) + len(test_y)}

def prepare_data(node_dict, test_perc, max_proteins=60000):
    basename = 'tmp/'+params['cluster_name']
    train_x_name = basename + '.train_x.obj'
    train_y_name = basename + '.train_y.obj'
    test_x_name = basename + '.test_x.obj'
    test_y_name = basename + '.test_y.obj'

    if all([path.exists(p) for p in [train_x_name, train_y_name, test_x_name, test_y_name]]):
        return True
    else:
        print('Preparing', basename)
        test_go_set = params['test_go_set']
        go_annotations = params['go_annotations']
        all_proteins = set()
        for go in test_go_set:
            annots = go_annotations[go]
            print(go, len(annots))
            all_proteins.update(annots)
        
        print(len(all_proteins), 'proteins')
        protein_list = sorted(all_proteins)
        if len(protein_list) > max_proteins:
            protein_list = sample(protein_list, max_proteins)

        print('Loading features')
        features, local_labels = make_dataset('input/traintest', protein_list, test_go_set)
        train_ids, train_x, train_y, test_ids, test_x, test_y = split_train_test(
            protein_list, features, local_labels)

        to_save = [(train_x_name, train_x), (train_y_name, train_y),
                (test_x_name, test_x), (test_y_name, test_y),]
        if not path.exists('tmp'):
            run_command(['mkdir', 'tmp'])
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))
        
        return True

def sample_train_test(traintest_path, features, test_perc, max_proteins=60000):
    base_dir = traintest_path.replace('.parquet', '_test'+str(test_perc)+"_features-"+'-'.join(features))
    
    train_x_name = base_dir + '/train_x.obj'
    train_y_name = base_dir + '/train_y.obj'
    test_x_name = base_dir + '/test_x.obj'
    test_y_name = base_dir + '/test_y.obj'

    all_paths = [train_x_name, train_y_name, test_x_name, test_y_name]

    if all([path.exists(p) for p in all_paths]):
        return True
    else:
        print('Preparing', base_dir)
        run_command(['mkdir -p', base_dir])
        cols_to_use = ['id'] + features + ['labels']
        traintest = load_columns_from_parquet(traintest_path, cols_to_use)
        if len(traintest) > max_proteins:
            print(traintest_path, 'is too large, sampling down')
            traintest = traintest.sample(fraction=(max_proteins/len(traintest)), 
                shuffle=True, seed=1337)
        splited = split_train_test_polars(traintest, test_perc)
        train_ids, train_x, train_y, test_ids, test_x, test_y = splited

        train_x_np = []
        for col in train_x.columns:
            train_x_np.append((col, train_x[col].to_numpy()))
        train_y_np = train_y['labels'].to_numpy()

        test_x_np = []
        for col in test_x.columns:
            test_x_np.append((col, test_x[col].to_numpy()))
        test_y_np = test_y['labels'].to_numpy()
        
        to_save = [(train_x_name, train_x_np), (train_y_name, train_y_np),
                (test_x_name, test_x_np), (test_y_name, test_y_np),]
        if not path.exists('tmp'):
            run_command(['mkdir', 'tmp'])
        for p, obj in to_save:
            dump(obj, open(p, 'wb'))

        open(base_dir+'/train_ids.txt', 'w').write('\n'.join(train_ids))
        open(base_dir+'/test_ids.txt', 'w').write('\n'.join(test_ids))
        
        return True

def train_node(params, features, max_proteins=60000):
    node = params['node']
    params_dict = params['params_dict']

    traintest_path = node['traintest_path']
    base_dir = traintest_path.replace('.parquet', 
        '_test'+str(params['test_perc'])+"_features-"+'-'.join(features))
    #print('loading', base_dir)
    
    train_x_name = base_dir + '/train_x.obj'
    train_y_name = base_dir + '/train_y.obj'
    test_x_name = base_dir + '/test_x.obj'
    test_y_name = base_dir + '/test_y.obj'

    train_x = load(open(train_x_name, 'rb'))
    train_y = load(open(train_y_name, 'rb'))
    test_x = load(open(test_x_name, 'rb'))
    test_y = load(open(test_y_name, 'rb'))

    annot_model, stats = makeMultiClassifierModel(train_x, train_y, test_x, test_y, 
        params_dict)
    #print(params['cluster_name'], stats)

    return annot_model, stats

def predict_with_model(nodes, experiment_dir):
    print('Loading validation')
    val_protein_list = open('input/validation/ids.txt', 'r').read().split('\n')
    print('Loading features')
    val_features, labels, annotations = load_dataset_from_dir('input/validation', val_protein_list)
    for feature_name, feature_vec in val_features:
        print('\t', feature_name, len(feature_vec), len(feature_vec[0]))
    val_features = x_to_np(val_features)
    val_x = [x for name, x in val_features]

    roc_auc_scores = []
    all_targets = []
    all_probas = [[] for _ in range(len(val_protein_list))]
    for mod_name, data in nodes.items():
        val_tsv = experiment_dir+'/'+mod_name+'.val.tsv'
        output = open(val_tsv, 'w')
        annot_model, targets = data
        all_targets += targets
        output.write('protein\ttaxid\t'+ '\t'.join(targets)+'\n')
        #print('Creating go label one hot encoding')
        val_y = create_labels_matrix(labels, val_protein_list, targets)
        #print('\t', val_y.shape)
    
        print('Validating')
        val_y_pred = annot_model.predict(val_x)
        #roc_auc_score = metrics.roc_auc_score(val_y, val_y_pred)
        #print(roc_auc_score)
        #roc_auc_scores.append(roc_auc_score)

        for i in range(len(val_protein_list)):
            predicted_probs = [x for x in val_y_pred[i]]
            predicted_probs_str = [str(x) for x in predicted_probs]
            all_probas[i] += predicted_probs
            output.write(val_protein_list[i]+'\t' + '\t'.join(predicted_probs_str)+'\n')
        output.close()
    
    big_table_path = experiment_dir+'/validation.tsv'
    output = open(big_table_path, 'w')
    output.write('protein\ttaxid\t'+ '\t'.join(all_targets)+'\n')
    for i in range(len(val_protein_list)):
        predicted_probs_str = [str(x) for x in all_probas[i]]
        output.write(val_protein_list[i]+'\t' + '\t'.join(predicted_probs_str)+'\n')
    output.close()

    return big_table_path

class MetaheuristicTest():

    def __init__(self, name, params, features, pop) -> None:
        self.nodes = params

        '''self.problem_constrained = {
            "obj_func": self.objective_func,
            "bounds": PARAM_TRANSLATOR.to_bounds(),
            "minmax": "max",
            "log_to": "file",
            "log_file": "result.log",         # Default value = "mealpy.log"
        }'''

        params_dict = {k: {k2: v2 for k2, v2 in v.items()} for k, v in param_bounds.items()}
        plm_base_params = params_dict['plm']
        for feature_name in features:
            feature_len = plm_sizes[feature_name]
            params_dict[feature_name] = {
                "l1_dim": [int(plm_base_params["l1_dim"][0]*feature_len), 
                           int(plm_base_params["l1_dim"][1]*feature_len)],
                "l2_dim": [int(plm_base_params["l2_dim"][0]*feature_len), 
                           int(plm_base_params["l2_dim"][1]*feature_len)],
                "dropout_rate": plm_base_params['dropout_rate'],
                "leakyrelu_1_alpha": plm_base_params['leakyrelu_1_alpha']
            }
        del params_dict['plm']
        self.features = features
        self.new_param_translator = ProblemTranslator(params_dict)
        self.heuristic_model = RandomSearchMetaheuristic(name, self.new_param_translator, pop,
            n_jobs=3)
    
    def objective_func(self, solution):
        #print('objective_func', file=sys.stderr)
        new_params_dict = self.new_param_translator.decode(solution)
        
        for node in self.nodes:
            node['params_dict'] = new_params_dict
        '''n_procs = config['training_processes']
        if n_procs > len(self.nodes):
            n_procs = len(self.nodes)'''

        #print('getting roc_auc s', file=sys.stderr)
        roc_aucs = [train_node(node, self.features)[1]['ROC AUC'] for node in self.nodes]
        #print(roc_aucs, file=sys.stderr)
        mean_training_roc = np.mean(roc_aucs)
        min_roc_auc = min(roc_aucs)
        std = np.std(roc_aucs)
        #print('objective_func finish', file=sys.stderr)
        return mean_training_roc, min_roc_auc, std

    def test(self):
        best_solution, best_fitness, report = self.heuristic_model.run_tests(
            self.objective_func, gens=2, top_perc=0.5)
        solution_dict = self.new_param_translator.decode(best_solution)
        #print(json.dumps(solution_dict, indent=4))
        print(best_fitness)

        results = {}
        rocs = []
        for node in self.nodes:
            node['params_dict'] = solution_dict
            annot_model, stats = train_node(node, self.features)
            print('Validating')
            val_path = node['node']['val_path']
            val_df = pl.read_parquet(val_path)
            val_x_np = []
            for col in self.features:
                val_x_np.append(val_df[col].to_numpy())
            val_df_y_np = val_df['labels'].to_numpy()
            val_y_pred = annot_model.predict(val_x_np, verbose=0)
            roc_auc_score = metrics.roc_auc_score(val_df_y_np, val_y_pred)
            acc = np.mean(keras.metrics.binary_accuracy(val_df_y_np, val_y_pred).numpy())
            print(roc_auc_score)
            results[node['node_name']] = {
                'test': stats,
                'validation': {
                    'roc_auc_score': float(roc_auc_score),
                    'acc': float(acc),
                    'val_x': len(val_df)
                }
            }
            if roc_auc_score == roc_auc_score:
                rocs.append(roc_auc_score)
        results['roc_auc_mean'] = np.mean(rocs)
        results['roc_auc_min'] = min(rocs)
        results['params_dict'] = solution_dict

        return results, best_fitness, report
