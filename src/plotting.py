from glob import glob
import json
from os import path
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
from decimal import Decimal

from parsing import load_gens_df, load_final_solutions, load_solutions, load_taxa_populations

model_colors = {
    'ankh_base': 'red', 'ankh_large': 'darkred', 
    'esm2_t6': '#8FF259', 'esm2_t12': '#43D984', 
    'esm2_t30': '#3F1C34', 'esm2_t33': '#FFF955', 
    'esm2_t36': '#AD00B0', 
    'prottrans': 'blue'
}

def plot_gens_evol(gens_df, output_path, metric_to_plot):
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    all_gens = set()
    for model_name, model_rows in gens_df.groupby('model'):
        x = []
        y = []
        for gen_n, model_gen_rows in model_rows.groupby('gen'):
            max_val = model_gen_rows[metric_to_plot].max()
            x.append(gen_n)
            y.append(max_val)
        all_gens.update(x)
        if not model_name in model_colors:
            names = model_name.split('-')
            if len(names) == 2:
                ax.plot(x, y, label=names[0], linewidth=6, alpha=0.8, color=model_colors[names[0]])
                ax.plot(x, y, label=names[1], linewidth=4, alpha=0.7, color=model_colors[names[1]])
        else:
            ax.plot(x, y, label=model_name, linewidth=6, alpha=0.7, color=model_colors[model_name])
        
    all_gens = [int(g) for g in sorted(all_gens)]
    ax.set_xticks(all_gens, [str(g) for g in all_gens])
    ax.set_xlim(min(all_gens), max(all_gens))
    ax.legend()
    ax.set_title('Metaheuristics Evolution - ' + metric_to_plot)
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=200)
    except Exception as err:
        pass

def iterative_gens_draw(benchmark_path, prev_n_gens=0):
    gens_df = load_gens_df(benchmark_path)
    if len(gens_df) > prev_n_gens:
        for m in ['f1_score_w_06', 'ROC AUC W', 'precision_score_w_06', 'fitness']:
            gens_plot_path = benchmark_path +'/evol-'+m+'.png'
            plot_gens_evol(gens_df, gens_plot_path, m)
        gens_df.to_csv(benchmark_path + '/all_gens.csv', sep=',')
    return len(gens_df)

def plot_metrics(benchmark_path, final=True):
    if final:
        solutions = load_final_solutions(benchmark_path)
        
        metrics = [(v['solution'], v['metrics']['ROC AUC W']) 
                   for k, v in solutions.items()]
        
        values = {
            k: [] for k in metrics[0][0].keys()
        }
        
        for m, roc_auc in metrics:
            #print(roc_auc, m)
            for k, v in m.items():
                values[k].append((roc_auc, v))
    else:
        solutions = load_solutions(benchmark_path)
        param_names = solutions[list(solutions.keys())[0]]['population_best'][0]['params'].keys()
        values = {
            k: [] for k in param_names
        }
        for name, data in solutions.items():
            pop = data['population_best']
            print(pop[0])
            last_i = len(pop)-1
            for i, p in enumerate(pop):
                if i == last_i:
                    for m_name, m_value in p['params'].items():
                        values[m_name].append((p['roc'], m_value, True))
                else:
                    for m_name, m_value in p['params'].items():
                        values[m_name].append((p['roc'], m_value, False))
    
    for m, vs in values.items():
        #print(m, vs)
        roc_auc = [x for x, y, _ in vs]
        metric_vals = [y for x, y, _ in vs]
        
        roc_auc_best = [x for x, y, b in vs if b]
        metric_vals_best = [y for x, y, b in vs if b]
    
        plot_path = benchmark_path + '/metric_' + m + '.png'
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        #print(metric_vals)
        #print(roc_auc)
        ax.scatter(metric_vals, roc_auc, s=30, alpha=0.5)
        ax.scatter(metric_vals_best, roc_auc_best, s=120, 
                   alpha=1.0, marker='*')
        #for i, txt in enumerate(names):
        #    ax.annotate(txt.upper().replace('_', ' '), (precision[i], roc[i]), ha='center', va='bottom')
        ax.set_xlabel(m)
        ax.set_ylabel('ROC AUC Weighted')
        ax.set_title(m + ' x ROC AUC')
        if 'learning_rate' in m:
            min_val = min(metric_vals)
            max_val = max(metric_vals)
            space = (max_val - min_val) / 4
            new_tick_vals = [min_val, min_val+space, min_val+space*2, min_val+space*3, max_val]
            new_ticks = ['%.2E' % Decimal(str(i)) for i in new_tick_vals]
            #print(new_tick_vals, new_ticks)
            
            ax.set_xticks(new_tick_vals, new_ticks)
        else:
            pass
        fig.tight_layout()
        
        fig.savefig(plot_path, dpi=120)
    
    l1dim = values['plm_l1_dim']
    l2dim = values['plm_l2_dim']
    assert len(l1dim) == len(l2dim)
    
    x = []
    x_best = []
    y = []
    y_best = []
    s1 = []
    s_best = []
    
    for l1_metrics, l2_metrics in zip(l1dim, l2dim):
        if l1_metrics[2]:
            x_best.append(l1_metrics[1])
            y_best.append(l2_metrics[1])
            s_best.append(100+l2_metrics[0]*l2_metrics[0]*60)
        else:
            x.append(l1_metrics[1])
            y.append(l2_metrics[1])
            s1.append(40+l2_metrics[0]*l2_metrics[0]*30)
    
    plot_path = benchmark_path + '/l1xl2.png'
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(x, y, s=s1, alpha=0.5)
    ax.scatter(x_best, y_best, s=s_best, 
               alpha=1.0, marker='*')
    #for i, txt in enumerate(names):
    #    ax.annotate(txt.upper().replace('_', ' '), (precision[i], roc[i]), ha='center', va='bottom')
    ax.set_xlabel('L1 DIM')
    ax.set_ylabel('L2 DIM')
    #ax.set_title(m + ' x ROC AUC')
    fig.tight_layout()
    
    fig.savefig(plot_path, dpi=120)

def plot_taxon_metrics(benchmark_path):
    populations = load_taxa_populations(benchmark_path)
    param_names = set()
    for s in list(populations.keys()):
        print(s)
        print(len(populations[s]['population']))
        print(len(populations[s]['population_best']))
    for s in list(populations.keys()):
        param_names.update(populations[s]['population'][0]['params'].keys())
    #param_names = solutions[list(solutions.keys())[0]]['population_best'][0]['params'].keys()
    print("Parameters to plot:", param_names)
    values = {
        k: [] for k in param_names
    }
    for name, data in populations.items():
        pop = data['population']
        print(pop[0])
        last_i = len(pop)-1
        good_quality_start = len(pop)-32
        if good_quality_start < 0:
            good_quality_start = 0
        for i, p in enumerate(pop):
            if not p['is_best']:
                if i < good_quality_start:
                    label = 'Bad'
                else:
                    label = 'Good'
            else:
                label = 'Best'
            
            for m_name, m_value in p['params'].items():
                values[m_name].append((p['auprc'], m_value, label))
    
    for m, vs in values.items():
        #print(m, vs)

        roc_auc_bad = [x for x, y, l in vs if l == 'Bad']
        metric_vals_bad = [y for x, y, l in vs if l == 'Bad']

        roc_auc = [x for x, y, l in vs if l == 'Good']
        metric_vals = [y for x, y, l in vs if l == 'Good']
        
        roc_auc_best = [x for x, y, l in vs if l == 'Best']
        metric_vals_best = [y for x, y, l in vs if l == 'Best']
    
        plot_path = benchmark_path + '/metric_' + m + '.png'
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        #print(metric_vals)
        #print(roc_auc)
        ax.scatter(metric_vals_bad, roc_auc_bad, s=10, alpha=0.4, label='Bad')
        ax.scatter(metric_vals, roc_auc, s=40, alpha=0.6, label='Good')
        ax.scatter(metric_vals_best, roc_auc_best, s=180, 
                   alpha=1.0, marker='*', label='Gen. Best')
        #for i, txt in enumerate(names):
        #    ax.annotate(txt.upper().replace('_', ' '), (precision[i], roc[i]), ha='center', va='bottom')
        ax.set_xlabel(m)
        ax.set_ylabel('auprc Weighted')
        ax.set_title(m + ' x auprc')
        if 'learning_rate' in m:
            min_val = min(metric_vals)
            max_val = max(metric_vals)
            space = (max_val - min_val) / 4
            new_tick_vals = [min_val, min_val+space, min_val+space*2, min_val+space*3, max_val]
            new_ticks = ['%.2E' % Decimal(str(i)) for i in new_tick_vals]
            #print(new_tick_vals, new_ticks)
            
            ax.set_xticks(new_tick_vals, new_ticks)
        else:
            pass
        fig.tight_layout()
        
        fig.savefig(plot_path, dpi=120)
    
    '''l1dim = values['plm_l1_dim']
    l2dim = values['plm_l2_dim']
    assert len(l1dim) == len(l2dim)
    
    x = []
    x_best = []
    y = []
    y_best = []
    s1 = []
    s_best = []
    
    for l1_metrics, l2_metrics in zip(l1dim, l2dim):
        if l1_metrics[2]:
            x_best.append(l1_metrics[1])
            y_best.append(l2_metrics[1])
            s_best.append(100+l2_metrics[0]*l2_metrics[0]*60)
        else:
            x.append(l1_metrics[1])
            y.append(l2_metrics[1])
            s1.append(40+l2_metrics[0]*l2_metrics[0]*30)
    
    plot_path = benchmark_path + '/l1xl2.png'
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(x, y, s=s1, alpha=0.5)
    ax.scatter(x_best, y_best, s=s_best, 
               alpha=1.0, marker='*')
    
    ax.set_xlabel('L1 DIM')
    ax.set_ylabel('L2 DIM')
    
    fig.tight_layout()
    
    fig.savefig(plot_path, dpi=120)'''

def plot_final_solution_performance(benchmark_path):
    solutions = load_final_solutions(benchmark_path)

    precision = []
    roc = []
    names = []

    for name, v in solutions.items():
        precision.append(v['metrics']['AUPRC'])
        roc.append(v['metrics']['ROC AUC W'])
        names.append(name)

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(precision, roc, s=160)
    for i, txt in enumerate(names):
        ax.annotate(txt.upper().replace('_', ' '), (precision[i], roc[i]), ha='center', va='bottom')
    ax.set_xlabel('AUPRC')
    ax.set_ylabel('ROC AUC Weighted')
    ax.set_title('M.F. Classification Performance of PLMs')
    fig.tight_layout()
    
    fig.savefig(benchmark_path+'/model_performance.png', dpi=120)

def draw_swarm_panel(full_swarm_exp_dir: str, output_dir: str):
    node_dirs = glob(full_swarm_exp_dir + '/Level-*')
    node_dicts = [x+'/exp_params.json' for x in node_dirs]
    node_results = [x+'/exp_results.json' for x in node_dirs]
    
    n_proteins = []
    node_names = []
    auprc_ws = []
    roc_auc_ws = []
    node_levels = []
    node_pos_in_level = []
    n_labels = {}
    for exp_params, exp_results in zip(node_dicts, node_results):
        if not path.exists(exp_params):
            exp_params = exp_params.replace('exp_params.json', 'standard_params.json')
        params = json.load(open(exp_params, 'r'))

        node_names.append(params['node_name'])
        n_proteins.append(len(params['node']['id']))
        node_levels.append(int(node_names[-1].split('_')[0].split('-')[1]))
        n_labels[node_names[-1]] = len(params['node']['go'])

        std_results = exp_results.replace('exp_results.json', 'standard_results.json')
        if path.exists(exp_results):
            results = json.load(open(exp_results, 'r'))
            auprc_ws.append(results['validation']['AUPRC W'])
            roc_auc_ws.append(results['validation']['ROC AUC W'])
        elif path.exists(std_results):
            results = json.load(open(std_results, 'r'))
            auprc_ws.append(results['validation']['AUPRC W'])
            roc_auc_ws.append(results['validation']['ROC AUC W'])
        else:
            #print('No results at', exp_results)
            auprc_ws.append(None)
            roc_auc_ws.append(None)
    
    y_positions = []
    for i in range(len(node_levels)):
        level_names = [n for j, n in enumerate(node_names) if node_levels[j] == node_levels[i]]
        node_name = node_names[i]
        names_sorted = sorted(level_names, 
            key=lambda n: int(n.split('Freq-')[1].split('-')[0]), reverse=True)
        pos_in_level = names_sorted.index(node_name)
        node_pos_in_level.append(pos_in_level)
        if pos_in_level % 2 == 0:
            y_positions.append(node_levels[i]-0.125)
        else:
            y_positions.append(node_levels[i]+0.125)
    x_positions = []
    for index in range(len(y_positions)):
        x_positions.append(node_pos_in_level[index]*0.25)
    deepest_level = max(node_levels)
    levels_width = max(node_pos_in_level)
    auprc_min = min([x for x in auprc_ws if x is not None])
    proteins_min = min(n_proteins)
    proteins_max = max(n_proteins)
    labels_min = min([x for x in n_labels.values()])
    labels_max = max([x for x in n_labels.values()])
    cmap = cm.winter
    m1 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=auprc_min, vmax=1.0), 
        cmap=cmap)
    m2 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=proteins_min, vmax=proteins_max), 
        cmap=cm.seismic)
    m3 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=labels_min, vmax=labels_max), 
        cmap=cm.seismic)
    plot_w = 5.5
    plot_h = 11
    fig1, ax1 = plt.subplots(1,1, figsize=(plot_w,plot_h))
    fig2, ax2 = plt.subplots(1,1, figsize=(plot_w,plot_h))
    fig3, ax3 = plt.subplots(1,1, figsize=(plot_w,plot_h))

    metrics_labels = ['AUPRC Weighted', 'Proteins for Train/test', 'GO IDs Predicted']
    axes = [ax1, ax2, ax3]
    figs = [fig1, fig2, fig3]
    color_maps = [m1, m2, m3]

    for ax in axes:
        ax.scatter(x_positions, y_positions, c='white')

    for index in range(len(x_positions)):
        x_pos = x_positions[index]
        y_pos = y_positions[index]
        label = node_names[index]
        n_go_ids = n_labels[label]
        traintest_proteins = n_proteins[index]
        current_auprcw = auprc_ws[index]
        
        node_color1 = m1.to_rgba(current_auprcw) if current_auprcw is not None else 'white'
        node_color2 = m2.to_rgba(traintest_proteins)
        node_color3 = m3.to_rgba(n_go_ids)
        
        node_pos = (x_pos, y_pos)
        node_w = 0.25
        node_h = 0.5
        node_circle1 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color1, edgecolor='black')
        ax1.add_patch(node_circle1)
        node_circle2 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color2, edgecolor='black')
        ax2.add_patch(node_circle2)
        node_circle3 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color3, edgecolor='black')
        ax3.add_patch(node_circle3)
    
    #ax.get_xaxis().set_visible(False)
    #ax.set_xscale('log', base=30)
    min_maxes = [(auprc_min, 1.0), 
        (proteins_min, proteins_max), 
        (labels_min, labels_max)]
    for fig, ax, label, m, min_max in zip(figs, axes, metrics_labels, color_maps, min_maxes):
        ax.set_ylabel("Gene Ontology Level", fontsize=14)
        #ax.set_xlabel("Models in Layer", fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([1,2,3,4,5,6,7])
        ax.set_xticks([])
        #ax.xaxis.tick_top()
        #ax.xaxis.set_label_position('top')

        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(label+"\nof Base Models", fontsize=16)
        metric_min, metric_max = min_max
        fourth = (metric_max - metric_min) / 4
        point_2 = metric_min + fourth
        point_3 = metric_min + fourth *2
        point_4 = metric_max - fourth
        cbar = fig.colorbar(m, ax=ax, label=label, fraction=0.09,
            ticks=[metric_min, point_2, point_3, point_4, metric_max])
        cbar.ax.set_yticklabels(
            cbar.ax.get_yticklabels(), rotation=90, ha='left', va='center')
        #cbar.ax.yaxis.label.set_rotation(45)
        fig.tight_layout()
        #plt.show()
        label_norm = label.lower().replace(' ', '_').replace('/', '_')
        output_path = output_dir + '/swarm_' + label_norm + '.png'
        fig.savefig(output_path, dpi=120)

if __name__ == '__main__':
    full_swarm_exp_dir = sys.argv[1]
    draw_swarm_panel(full_swarm_exp_dir, 'img/')
    
    