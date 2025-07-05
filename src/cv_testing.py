#%load_ext autoreload
#%autoreload 2
from datetime import datetime
import json
from multiprocessing import Pool
from os import path
import os
import sys
import numpy as np
from pickle import load, dump
import polars as pl

print("New thread", file=sys.stderr)

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# from metaheuristics import ProblemTranslator, GeneticAlgorithm, RandomSearchMetaheuristic
from create_dataset import Dataset, find_latest_dataset
from dimension_db import DimensionDB
from cross_validation import split_train_test_n_folds, BasicEnsemble
from node_factory import makeMultiClassifierModel
# from node_factory import create_params_for_features, sample_train_test
from util_base import run_command
# from base_benchmark import BaseBenchmarkRunner, run_validation

top_feature_list = ["taxa_256", "ankh_base", "esm2_t33"]

def test_benchmarking_node(datasets_dir, min_proteins_per_mf, dimension_db_release_n,
        val_perc, dimension_db_releases_dir):
    dataset_type = "taxon_benchmark_cv"
    matching_dataset_path = find_latest_dataset(
        datasets_dir,
        dataset_type,
        min_proteins_per_mf,
        dimension_db_release_n,
        val_perc,
    )
    if matching_dataset_path is not None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dimension_db = DimensionDB(
            dimension_db_releases_dir, dimension_db_release_n, new_downloads=True
        )
        dataset = Dataset(
            dimension_db=dimension_db,
            min_proteins_per_mf=min_proteins_per_mf,
            dataset_type=dataset_type,
            val_perc=val_perc,
        )
    print("Nodes in dataset:", dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)

    node = list(dataset.go_clusters.values())[0]
    node_name = list(dataset.go_clusters.keys())[0]

    success_split = split_train_test_n_folds(node['traintest_path'], top_feature_list)

    exp_name = "cv_val-" + "-".join(top_feature_list)

    params_dict = {
        "ankh_base": {
            "dropout_rate": 0.7273763547614519,
            "l1_dim": 661,
            "l2_dim": 1668,
            "leakyrelu_1_alpha": 0.026733205039049884,
        },
        "esm2_t33": {
            "dropout_rate": 0.7956844174771458,
            "l1_dim": 636,
            "l2_dim": 433,
            "leakyrelu_1_alpha": 0.05286754528717738,
        },
        "final": {
            "batch_size": 130,
            "dropout_rate": 0.35154186929241227,
            "epochs": 39,
            "final_dim": 508,
            "learning_rate": 0.0006536915731422689,
            "patience": 9,
        },
        "taxa_256": {
            "dropout_rate": 0.3545169266005104,
            "l1_dim": 100,
            "l2_dim": 68,
            "leakyrelu_1_alpha": 0.5447727242633195,
        },
    }
    n_folds = 5
    
    #'metric_name': "fitness",
    #'metric_name2': 'f1_score_w_06',

    traintest_path = node['traintest_path']
    base_dir = traintest_path.replace('.parquet', 
        '_folds'+str(n_folds)+"_features-"+'-'.join(top_feature_list))
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


    val_path = node['val_path']
    go_labels = node['go']
    val_df = pl.read_parquet(val_path)
    val_x_np = []
    for col in top_feature_list:
        val_x_np.append(val_df[col].to_numpy())
    val_df_y_np = val_df['labels'].to_numpy()

    from sklearn import metrics

    for model in models + [annot_model]:
        val_y_pred = model.predict(val_x_np, verbose=0)
        roc_auc_score_mac = metrics.roc_auc_score(val_df_y_np, val_y_pred, average='macro')
        roc_auc_score_w = metrics.roc_auc_score(val_df_y_np, val_y_pred, average='weighted')
        auprc_mac = metrics.average_precision_score(val_df_y_np, val_y_pred)
        auprc_w = metrics.average_precision_score(val_df_y_np, val_y_pred, average='weighted')
        val_stats = {'ROC AUC': float(roc_auc_score_mac),
            'ROC AUC W': float(roc_auc_score_w),
            'AUPRC': float(auprc_mac),
            'AUPRC W': float(auprc_w)}
        print(val_stats)

def test_nodes_from_full_clustering(datasets_dir, min_proteins_per_mf, 
        dimension_db_release_n, val_perc, dimension_db_releases_dir):
    dataset_type = "full_swarm"
    matching_dataset_path = find_latest_dataset(
        datasets_dir,
        dataset_type,
        min_proteins_per_mf,
        dimension_db_release_n,
        val_perc,
    )
    if matching_dataset_path is not None:
        dataset = Dataset(dataset_path=matching_dataset_path)
    else:
        dimension_db = DimensionDB(
            dimension_db_releases_dir, dimension_db_release_n, new_downloads=True
        )
        dataset = Dataset(
            dimension_db=dimension_db,
            min_proteins_per_mf=min_proteins_per_mf,
            dataset_type=dataset_type,
            val_perc=val_perc,
        )
    print("Nodes in dataset:", dataset.go_clusters.keys())
    if dataset.new_dataset:
        dataset.save(datasets_dir)

    exp_name = "swarm_val-" + "-".join(top_feature_list)

    params_dict = {
        "ankh_base": {
            "dropout_rate": 0.7273763547614519,
            "l1_dim": 661,
            "l2_dim": 1668,
            "leakyrelu_1_alpha": 0.026733205039049884,
        },
        "esm2_t33": {
            "dropout_rate": 0.7956844174771458,
            "l1_dim": 636,
            "l2_dim": 433,
            "leakyrelu_1_alpha": 0.05286754528717738,
        },
        "final": {
            "batch_size": 130,
            "dropout_rate": 0.35154186929241227,
            "epochs": 39,
            "final_dim": 508,
            "learning_rate": 0.0006536915731422689,
            "patience": 9,
        },
        "taxa_256": {
            "dropout_rate": 0.3545169266005104,
            "l1_dim": 100,
            "l2_dim": 68,
            "leakyrelu_1_alpha": 0.5447727242633195,
        },
    }
    n_folds = 5

    node_name = 'Level-1_Freq-42-289_N-7'
    node = dataset.go_clusters[node_name]
    traintest_path = node['traintest_path']

    success_split = split_train_test_n_folds(traintest_path, top_feature_list)
    base_dir = traintest_path.replace('.parquet', 
        '_folds'+str(n_folds)+"_features-"+'-'.join(top_feature_list))

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
        print(stats)

    annot_model = BasicEnsemble(models, stats_dicts)
    print(annot_model.stats)

    return dataset, node, annot_model, params_dict

if __name__ == "__main__":
    dimension_db_release_n = "1"
    dimension_db_releases_dir = path.expanduser("~/data/protein_dimension_db/")
    datasets_dir = path.expanduser("~/data/mf_swarm_datasets/")
    benchmark_dir = "tmp"
    local_dir = benchmark_dir + "/cv_validation"

    run_command(["mkdir -p", local_dir])

    min_proteins_per_mf = 40
    val_perc = 0.15
    #test_benchmarking_node(datasets_dir, min_proteins_per_mf, dimension_db_release_n,
    #    val_perc, dimension_db_releases_dir, )

    dataset, node, annot_model, params_dict = test_nodes_from_full_clustering(
        datasets_dir, min_proteins_per_mf, 
        dimension_db_release_n, val_perc, dimension_db_releases_dir)
    
