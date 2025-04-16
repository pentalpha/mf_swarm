import matplotlib.pyplot as plt
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
    
if __name__ == '__main__':
    benchmark_path = '/home/pita/experiments/base_benchmark_4'
    #benchmark_path = sys.argv[1]
    iterative_gens_draw(benchmark_path)
    
    