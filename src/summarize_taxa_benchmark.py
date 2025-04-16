

import json
import sys
from os import path
from glob import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from parsing import load_final_metrics, load_final_solutions
from plotting import model_colors, plot_taxon_metrics
from util_base import plm_sizes
from tqdm import tqdm

#Equivalent to summarize_pairs_benchmark.py, but for taxa benchmark + the final table from pairs benchmark
if __name__ == '__main__':
    taxa_benchmark_dir = sys.argv[1]
    plot_others = sys.argv[2] == 'True'
    if plot_others:
        parent_dir = path.dirname(taxa_benchmark_dir)
        others = glob(parent_dir+'/*_benchmark_*')
        others = [x for x in others if not path.basename(taxa_benchmark_dir) in x]
        print('Other tests to load:')
        print(parent_dir+'/*_benchmark_')
        print('\n'.join(others))
    else:
        others = []
    plot_taxon_metrics(taxa_benchmark_dir)
    #pairs_results = load_final_metrics(pairs_benchmark_dir)
    #pairs_results = {}
    taxon_results = load_final_metrics(taxa_benchmark_dir)
    if plot_others:
        more_results = [load_final_metrics(o) for o in tqdm(others)]
    else:
        more_results = []

    metrics_to_use = ['f1_score_w_05', 'ROC AUC', 'ROC AUC W', 
                      'precision_score_w_06', 'recall_score', 'Accuracy',
                      'quickness', 'fitness', 'AUPRC', 'AUPRC W', 
                      'WORST AUPRC W', 'BEST AUPRC W', 
                      'WORST ROC AUC W', 'BEST ROC AUC W',
                      'WORST AUPRC', 'BEST AUPRC', 
                      'WORST ROC AUC', 'BEST ROC AUC']
    metric_pretty_names = ['F1 Score (W)', 'ROC AUC Score', 'ROC AUC Score (W)', 
                        'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
                        'Quickness', 'Fitness', 'AUPRC Score', 'AUPRC Score (W)', 
                        'WORST AUPRC Score (W)', 'BEST AUPRC Score (W)', 
                        'WORST ROC AUC Score (W)', 'BEST ROC AUC Score (W)', 
                        'WORST AUPRC Score', 'BEST AUPRC Score', 
                        'WORST ROC AUC Score', 'BEST ROC AUC Score']

    determinant_metric = 'AUPRC Score (W)'
    performances_dict = {}

    for result_dict in [taxon_results] + more_results:
        for name, v in result_dict.items():
            if not (('esm2_t6' in name and '-' in name) or (plot_others and name == 'None')):
                new_perf = {'model': name}
                for m, m2 in zip(metrics_to_use, metric_pretty_names):
                    new_perf[m2] = round(v['metrics'][m], 4)
                new_perf['Metaparameters'] = json.dumps(v['solution'])
                tp = 'TAXON + ANKH-BASE + PROTTRANS' if 'taxa' in name else 'PLM + PLM'
                if not '-' in name and not 'taxa' in name:
                    tp = 'PLM'
                new_perf['Type'] = tp

                feature_name_list = name.split('-')
                if tp == 'TAXON + ANKH-BASE + PROTTRANS':
                    feature_name_list += ['prottrans', 'ankh_base']
                total_len = 0
                for sub_n in feature_name_list:
                    if sub_n in plm_sizes:
                        total_len += plm_sizes[sub_n]
                    else:
                        print(name, tp, 'has no feature length defined')
                #if 'taxa' in name:
                #    total_len += plm_sizes['prottrans'] + plm_sizes['ankh_base']
                new_perf['Total Feature Length'] = total_len

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
    columns = ['model', 'Type',
        'ROC AUC Score', 'AUPRC Score', 
        'AUPRC Score (W)', 'ROC AUC Score (W)',
        'WORST AUPRC Score (W)', 'BEST AUPRC Score (W)', 
        'WORST ROC AUC Score (W)', 'BEST ROC AUC Score (W)',
        'WORST AUPRC Score', 'BEST AUPRC Score', 
        'WORST ROC AUC Score', 'BEST ROC AUC Score',
        'Quickness', 'Total Feature Length',
        'F1 Score (W)', 'Precision Score (W)', 'Recall Score', 'Binary Accuracy',
        'Fitness', 'Metaparameters']
    performances_df = pd.DataFrame(performances)
    performances_df = performances_df[columns]
    performances_df.to_csv(taxa_benchmark_dir + '/benchmark.tsv', sep='\t', index=False)

    metric_pairs = [('AUPRC Score (W)', 'ROC AUC Score (W)'), 
                    ('AUPRC Score', 'ROC AUC Score'),
                    ('BEST AUPRC Score (W)', 'WORST AUPRC Score (W)'),
                    ('BEST AUPRC Score', 'WORST AUPRC Score')]

    type_to_color = {
        'TAXON + ANKH-BASE + PROTTRANS': 'red',
        'PLM + PLM': 'blue',
        'PLM': 'green'
    }

    type_to_marker = {
        'TAXON + ANKH-BASE + PROTTRANS': 'H',
        'PLM + PLM': 's',
        'PLM': '<'
    }

    min_feature_len = performances_df['Total Feature Length'].min()
    max_feature_len = performances_df['Total Feature Length'].max()
    norm = plt.Normalize(min_feature_len, max_feature_len)
    cmap = LinearSegmentedColormap.from_list("", ["red","violet","blue"])
    feature_len_range = max_feature_len - min_feature_len
    min_marker_size = 60
    marker_range = 360

    for metric_a, metric_b in metric_pairs:
        fig, ax_main = plt.subplots(1, 1, figsize=(13,6.5))
        axin2 = ax_main.inset_axes([0.62, 0.02, 0.37, 0.4], 
            xlim=(0.7695, 0.7809), ylim=(0.885,0.892), xticklabels=[], yticklabels=[])
        #axin2 = zoomed_inset_axes(ax_main, loc='lower right')
        for tp, color_name in type_to_color.items():
            a = []
            #a_small = []
            b = []
            #b_small = []
            colors_big = []
            #colors_small = []
            names = []
            sizes = []
            for _, row in performances_df.iterrows():
                if row['Type'] == tp:
                    name = row['model']
                    test_tp = row['Type']
                    a.append(row[metric_a])
                    
                    b.append(row[metric_b])
                    colors_big.append(row['Total Feature Length'])
                    '''if '-' in name:
                        a_small.append(row[metric_a])
                        b_small.append(row[metric_b])
                        colors_small.append(model_colors[name.split('-')[1]])'''
                    feature_len = (row['Total Feature Length'] - min_feature_len) / feature_len_range
                    #sizes.append(int(min_marker_size + feature_len*marker_range))
                    sizes.append(200)
                    names.append(name)

        
            #ax_main = axes[0]
            ax_main.scatter(a, b, s=sizes, c=colors_big, label=tp, cmap=cmap, marker=type_to_marker[tp], norm=norm)
            axin2.scatter(a, b, s=sizes, c=colors_big, label=tp, cmap=cmap, marker=type_to_marker[tp], norm=norm)
            #ax_main.scatter(a_small, b_small, s=70, c=colors_small)
            for i, txt in enumerate(names):
                if 'taxa' in txt and not 'profile' in txt:
                    txt = txt.replace('taxa', 'taxa_onehot')
                '''if 'taxa' in txt:
                    txt = 'ankh_base-prottrans-\n'+txt'''
                ax_main.annotate(txt.upper().replace('_', ' ').replace('-', '+'), 
                            (a[i], b[i]), 
                            ha='center', va='bottom', fontsize=8, alpha=0.9)
                new_txt = txt.upper().replace('_', ' ').replace('-', '+')
                va2 = 'bottom'
                vertical_diff=0
                if new_txt == 'ANKH LARGE':
                    va2 = 'top'
                if new_txt == 'ESM2 T33+PROTTRANS':
                    va2 = 'top'
                axin2.annotate(new_txt, 
                            (a[i], b[i]+vertical_diff), 
                            ha='center', va=va2, fontsize=8, alpha=0.9)
        ax_main.set_xlabel(metric_a)
        ax_main.set_ylabel(metric_b)
        ax_main.set_title('M.F. Classification Performance Using Different Features')
        ax_main.legend()
        leg = ax_main.get_legend()
        leg.legend_handles[0].set_facecolor('#00000000')
        leg.legend_handles[0].set_edgecolor('black')
        leg.legend_handles[1].set_facecolor('#00000000')
        leg.legend_handles[1].set_edgecolor('black')
        leg.legend_handles[2].set_facecolor('#00000000')
        leg.legend_handles[2].set_edgecolor('black')
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), 
            ax=ax_main, label="Total Size of Input Features")

        ax_main.indicate_inset_zoom(axin2, edgecolor="black")

        fig.tight_layout()
        
        fig.savefig(
            taxa_benchmark_dir+'/model_performance-'+metric_a+'_'+metric_b+'.png',
            dpi=160)