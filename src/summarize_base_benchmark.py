

import sys

import pandas as pd


from parsing import load_final_solutions
from plotting import plot_final_solution_performance, plot_metrics

if __name__ == '__main__':
    #benchmark_path = '/home/pita/experiments/base_benchmark_4'
    benchmark_path = sys.argv[1]
    plot_final_solution_performance(benchmark_path)
    plot_metrics(benchmark_path, final=False)

    solutions = load_final_solutions(benchmark_path)

    performances = []
    metrics_to_use = ['AUPRC', 'f1_score_w_05', 'ROC AUC', 'ROC AUC W', 
                      'precision_score_w_06', 'recall_score', 'Accuracy']
    metric_pretty_names = ['AUPRC Score', 'F1 Score (W)', 'ROC AUC Score', 'ROC AUC Score (W)', 
                           'Precision Score (W)', 'Recall Score', 'Binary Accuracy']

    for name, v in solutions.items():
        new_perf = {'model': name}
        for m, m2 in zip(metrics_to_use, metric_pretty_names):
            new_perf[m2] = round(v['metrics'][m], 4)
            
        performances.append(new_perf)

    metric_weights = [('ROC AUC Score (W)', 3), ('F1 Score (W)', 4), ('Precision Score (W)', 1)]
    w_total = sum([w for m, w in metric_weights])

    performances.sort(key=lambda p: sum([p[m]*w for m, w in metric_weights]) / w_total, reverse=True)
    performances_df = pd.DataFrame(performances)
    performances_df.to_csv(benchmark_path + '/benchmark.tsv', sep='\t', index=False)

    