

import sys
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd

from parsing import load_final_metrics, load_final_solutions

if __name__ == '__main__':
    #benchmark_path = '/home/pita/experiments/base_benchmark_4'
    base_benchmark_dir = sys.argv[1]
    pairs_benchmark_dir = sys.argv[2]

    base_results = load_final_solutions(base_benchmark_dir)
    pairs_results = load_final_metrics(pairs_benchmark_dir)

    performances = []
    metrics_to_use = ['f1_score_w_05', 'ROC AUC', 'ROC AUC W', 
                      'precision_score_w_06', 'recall_score', 'Accuracy',
                      'quickness', 'fitness', 'AUPRC', 'AUPRC W']
    metric_pretty_names = ['F1 Score (W)', 'ROC AUC Score', 'ROC AUC Score (W)', 
                           'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
                           'Quickness', 'Fitness', 'AUPRC Score', 'AUPRC Score (W)']

    for name, v in base_results.items():
        new_perf = {'model': name}
        for m, m2 in zip(metrics_to_use, metric_pretty_names):
            new_perf[m2] = round(v['metrics'][m], 4)
            
        performances.append(new_perf)
    
    for name, v in pairs_results.items():
        new_perf = {'model': name}
        for m, m2 in zip(metrics_to_use, metric_pretty_names):
            new_perf[m2] = round(v[m], 4)
        #print(new_perf)
        performances.append(new_perf)

    metric_weights = [('ROC AUC Score (W)', 4), ('AUPRC Score', 4), ('AUPRC Score (W)', 2)]
    w_total = sum([w for m, w in metric_weights])

    performances.sort(
        key=lambda p: (p['AUPRC Score'], p['AUPRC Score (W)'], p['ROC AUC Score (W)']), 
        reverse=True)
    columns = ['model', 
        'ROC AUC Score', 'AUPRC Score', 
        'AUPRC Score (W)', 'ROC AUC Score (W)', 
        'Quickness',
        'F1 Score (W)', 'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
        'Fitness']
    performances_df = pd.DataFrame(performances)
    performances_df = performances_df[columns]
    performances_df.to_csv(pairs_benchmark_dir + '/benchmark.tsv', sep='\t', index=False)

    metric_pairs = [('AUPRC Score (W)', 'ROC AUC Score (W)'), ('AUPRC Score', 'ROC AUC Score')]

    for metric_a, metric_b in metric_pairs:
        a = []
        b = []
        names = []

        for _, row in performances_df.iterrows():
            name = row['model']
            a.append(row[metric_a])
            b.append(row[metric_b])
            names.append(name)

        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.scatter(a, b, s=160)
        for i, txt in enumerate(names):
            ax.annotate(txt.upper().replace('_', ' ').replace('-', '+'), 
                        (a[i], b[i]), 
                        ha='center', va='bottom', fontsize=7, alpha=0.8)
        ax.set_xlabel(metric_a)
        ax.set_ylabel(metric_b)
        ax.set_title('M.F. Classification Performance of PLMs')
        fig.tight_layout()
        
        fig.savefig(
            pairs_benchmark_dir+'/model_performance-'+metric_a+'_'+metric_b+'.png',
            dpi=160)

    