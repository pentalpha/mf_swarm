

import sys

import pandas as pd

from plotting import load_final_solutions, plot_final_solution_performance, plot_metrics

if __name__ == '__main__':
    #benchmark_path = '/home/pita/experiments/base_benchmark_4'
    pairs_benchmark_dir = sys.argv[1]
    pair_results_path = pairs_benchmark_dir + '/pair_results.json'
    plot_final_solution_performance(pairs_benchmark_dir)
    plot_metrics(pairs_benchmark_dir, final=False)

    solutions = load_final_solutions(pairs_benchmark_dir)

    performances = []
    metrics_to_use = ['f1_score_w_05', 'ROC AUC', 'ROC AUC W', 
                      'precision_score_w_06', 'recall_score', 'Accuracy']
    metric_pretty_names = ['F1 Score (W)', 'ROC AUC Score', 'ROC AUC Score (W)', 
                           'Precision Score (W)', 'Recall Score', 'Binary Accuracy']

    for name, v in solutions.items():
        new_perf = {'model': name}
        for m, m2 in zip(metrics_to_use, metric_pretty_names):
            new_perf[m2] = round(v['metrics'][m], 4)
            
        performances.append(new_perf)

    metric_weights = [('ROC AUC Score (W)', 3), ('F1 Score (W)', 4), ('Precision Score (W)', 1)]
    w_total = sum([w for m, w in metric_weights])

    performances.sort(key=lambda p: sum([p[m]*w for m, w in metric_weights]) / w_total, 
        reverse=True)
    performances_df = pd.DataFrame(performances)
    performances_df.to_csv(pairs_benchmark_dir + '/benchmark.tsv', sep='\t', index=False)

    