

import json
import sys
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
import matplotlib.patheffects as pe
from parsing import load_final_metrics, load_final_solutions
from plotting import model_colors, plot_taxon_metrics

#Equivalent to summarize_pairs_benchmark.py, but for taxa benchmark + the final table from pairs benchmark
if __name__ == '__main__':
    pairs_benchmark_path = sys.argv[1]
    taxa_benchmark_dir = sys.argv[2]
    if len(sys.argv) == 4:
        others = sys.argv[3].split(',')
    else:
        others = []
    
    plot_taxon_metrics(taxa_benchmark_dir)
    #pairs_results = load_final_metrics(pairs_benchmark_dir)
    pairs_results = {}
    taxon_results = load_final_metrics(taxa_benchmark_dir)
    more_results = [load_final_metrics(p) for p in others]

    metrics_to_use = ['f1_score_w_05', 'ROC AUC', 'ROC AUC W', 
                      'precision_score_w_06', 'recall_score', 'Accuracy',
                      'quickness', 'fitness', 'AUPRC', 'AUPRC W']
    metric_pretty_names = ['F1 Score (W)', 'ROC AUC Score', 'ROC AUC Score (W)', 
                           'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
                           'Quickness', 'Fitness', 'AUPRC Score', 'AUPRC Score (W)']

    determinant_metric = 'AUPRC Score (W)'
    performances_dict = {}

    for result_dict in [taxon_results, pairs_results] + more_results:
        for name, v in result_dict.items():
            if not ('esm2_t6' in name and '-' in name):
                new_perf = {'model': name}
                for m, m2 in zip(metrics_to_use, metric_pretty_names):
                    new_perf[m2] = round(v['metrics'][m], 4)
                new_perf['Metaparameters'] = json.dumps(v['solution'])
                tp = 'TAXON' if 'taxa' in name else 'PLM-PAIR'
                if not '-' in name:
                    tp = 'PLM'
                new_perf['Type'] = tp

                if not name in performances_dict:
                    performances_dict[name] = new_perf
                else:
                    highest_value = performances_dict[name][determinant_metric]
                    if new_perf[determinant_metric] > highest_value:
                        performances_dict[name] = new_perf
                #print(new_perf)
    performances = [v for k, v in performances_dict.items()]
    performances.sort(
        key=lambda p: (p['AUPRC Score (W)'], p['ROC AUC Score (W)'], p['AUPRC Score']), 
        reverse=True)
    columns = ['model', 
        'ROC AUC Score', 'AUPRC Score', 
        'AUPRC Score (W)', 'ROC AUC Score (W)', 
        'Quickness',
        'F1 Score (W)', 'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
        'Fitness', 'Metaparameters']
    performances_df = pd.DataFrame(performances)
    performances_df = performances_df[columns]
    performances_df.to_csv(taxa_benchmark_dir + '/benchmark.tsv', sep='\t', index=False)

    metric_pairs = [('AUPRC Score (W)', 'ROC AUC Score (W)'), 
                    ('AUPRC Score', 'ROC AUC Score')]


