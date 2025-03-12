

import json
import sys
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
import matplotlib.patheffects as pe
from parsing import load_final_metrics, load_final_solutions
from plotting import model_colors

if __name__ == '__main__':
    #benchmark_path = '/home/pita/experiments/base_benchmark_4'
    base_benchmark_dir = sys.argv[1]
    pairs_benchmark_dir = sys.argv[2]
    if len(sys.argv) == 4:
        others = sys.argv[3].split(',')
    else:
        others = []

    base_results = load_final_metrics(base_benchmark_dir)
    pairs_results = load_final_metrics(pairs_benchmark_dir)
    more_results = [load_final_metrics(p) for p in others]

    metrics_to_use = ['f1_score_w_05', 'ROC AUC', 'ROC AUC W', 
                      'precision_score_w_06', 'recall_score', 'Accuracy',
                      'quickness', 'fitness', 'AUPRC', 'AUPRC W']
    metric_pretty_names = ['F1 Score (W)', 'ROC AUC Score', 'ROC AUC Score (W)', 
                           'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
                           'Quickness', 'Fitness', 'AUPRC Score', 'AUPRC Score (W)']

    '''for name, v in base_results.items():
        new_perf = {'model': name}
        for m, m2 in zip(metrics_to_use, metric_pretty_names):
            new_perf[m2] = round(v['metrics'][m], 4)
            
        performances.append(new_perf)'''
    
    determinant_metric = 'AUPRC Score (W)'
    performances_dict = {}

    for result_dict in [base_results, pairs_results] + more_results:
        for name, v in result_dict.items():
            if not ('esm2_t6' in name and '-' in name):
                new_perf = {'model': name}
                for m, m2 in zip(metrics_to_use, metric_pretty_names):
                    new_perf[m2] = round(v['metrics'][m], 4)
                new_perf['Metaparameters'] = json.dumps(v['solution'])
                new_perf['Type'] = 'PLM-PAIR' if '-' in name else 'PLM'

                if not name in performances_dict:
                    performances_dict[name] = new_perf
                else:
                    highest_value = performances_dict[name][determinant_metric]
                    if new_perf[determinant_metric] > highest_value:
                        performances_dict[name] = new_perf
                #print(new_perf)
    performances = [v for k, v in performances_dict.items()]

    #metric_weights = [('ROC AUC Score (W)', 4), ('AUPRC Score', 4), ('AUPRC Score (W)', 2)]
    #w_total = sum([w for m, w in metric_weights])
    
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
    performances_df.to_csv(pairs_benchmark_dir + '/benchmark.tsv', sep='\t', index=False)

    metric_pairs = [('AUPRC Score (W)', 'ROC AUC Score (W)'), 
                    ('AUPRC Score', 'ROC AUC Score')]

    for metric_a, metric_b in metric_pairs:
        a = []
        a_small = []
        b = []
        b_small = []
        colors_big = []
        colors_small = []
        names = []
        for _, row in performances_df.iterrows():
            name = row['model']
            a.append(row[metric_a])
            
            b.append(row[metric_b])
            colors_big.append(model_colors[name.split('-')[0]])
            if '-' in name:
                a_small.append(row[metric_a])
                b_small.append(row[metric_b])
                colors_small.append(model_colors[name.split('-')[1]])
            names.append(name)

        fig, axes = plt.subplots(1, 2, figsize=(13,6.5))
        ax_main = axes[0]
        ax_main.scatter(a, b, s=170, c=colors_big)
        ax_main.scatter(a_small, b_small, s=70, c=colors_small)
        for i, txt in enumerate(names):
            ax_main.annotate(txt.upper().replace('_', ' ').replace('-', '+'), 
                        (a[i], b[i]), 
                        ha='center', va='bottom', fontsize=8, alpha=0.9)
        ax_main.set_xlabel(metric_a)
        ax_main.set_ylabel(metric_b)
        ax_main.set_title('M.F. Classification Performance of PLMs')

        ax_zoom = axes[1]
        original_x1, original_x2 = ax_main.get_xlim()
        original_y1, original_y2 = ax_main.get_ylim()

        new_x1 = original_x2 - (original_x2-original_x1)*0.25
        new_y1 = original_y2 - (original_y2-original_y1)*0.25

        a = []
        a_small = []
        b = []
        b_small = []
        colors_big = []
        colors_small = []
        names = []
        for _, row in performances_df.iterrows():
            if (row[metric_a] > new_x1 and row[metric_b] > new_y1):
                name = row['model']
                a.append(row[metric_a])
                
                b.append(row[metric_b])
                colors_big.append(model_colors[name.split('-')[0]])
                if '-' in name:
                    a_small.append(row[metric_a])
                    b_small.append(row[metric_b])
                    colors_small.append(model_colors[name.split('-')[1]])
                names.append(name)

        ax_zoom.scatter(a, b, s=400, c=colors_big)
        ax_zoom.scatter(a_small, b_small, s=120, c=colors_small)

        from adjustText import adjust_text

        texts = []
        for i, txt in enumerate(names):
            #ax_zoom.()
            new_txt = ax_zoom.text(a[i], b[i], txt.upper().replace('_', ' ').replace('-', '+'), 
                        ha='center', va='bottom', fontsize=11, alpha=1,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
            texts.append(new_txt)
        adjust_text(texts, 
                    #only_move={'points':'y', 'texts':'y'},
                    expand=(1.2, 1),
                    only_move='y',
                    arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        #ax_zoom.set_xlabel(metric_a)
        #ax_zoom.set_ylabel(metric_b)
        ax_zoom.set_title('Zoom At Best Performing Models')

        conn1 = ConnectionPatch((0.75, 0.75), (0.01, 0.01), 
                                'axes fraction', coordsB='axes fraction',
                                axesA=ax_main, axesB=ax_zoom)
        conn2 = ConnectionPatch((0.999, 0.999), (0.99, 0.99),
                                'axes fraction', coordsB='axes fraction',
                                axesA=ax_main, axesB=ax_zoom, zorder=-0.5)
        fig.add_artist(conn1)
        fig.add_artist(conn2)

        rect = Rectangle((new_x1, new_y1), original_x2-new_x1, original_y2-new_y1,
                         facecolor='#00000000', edgecolor='lightblue', fill=False, zorder=-1)
        ax_main.add_patch(rect)

        fig.tight_layout()
        
        fig.savefig(
            pairs_benchmark_dir+'/model_performance-'+metric_a+'_'+metric_b+'.png',
            dpi=160)

    