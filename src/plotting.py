from glob import glob
import json
from os import path
import matplotlib.pyplot as plt
from decimal import Decimal
import sys
import pandas as pd

from param_translator import ProblemTranslator
from util_base import plm_sizes

model_colors = {
    'ankh_base': 'red', 'ankh_large': 'darkred', 
    'esm2_t6': '#8FF259', 'esm2_t12': '#43D984', 
    'esm2_t30': '#3F1C34', 'esm2_t33': '#FFF955', 
    'esm2_t36': '#AD00B0', 
    'prottrans': 'blue'
}

def load_solutions(benchmark_path):
    val_jsons = glob(benchmark_path+'/*.json')
    vals = [json.load(open(p, 'r'))['validation'] for p in val_jsons]
    
    solution_jsons = [p.replace('.json', '/solution.json') for p in val_jsons]
    solutions = [json.load(open(p, 'r')) for p in solution_jsons]
    
    params_dict_custom = [json.load(open(p.replace('.json', '/params_dict_custom.json'), 'r')) 
                 for p in val_jsons]
    solution_translators = [ProblemTranslator(None, raw_values=d) for d in params_dict_custom]
    
    names = [path.basename(p).replace('.json', '') for p in val_jsons]
    
    solutions = {
        n: {'solution': s, 'metrics': v, 'translator': st}
        for v, s, n, st in zip(vals, solutions, names, solution_translators)
    }
    
    for n, data in solutions.items():
        data['solution']['plm'] = data['solution'][n]
        del data['solution'][n]
        
        for key_group_name in ['plm', 'final']:
            for key, v in data['solution'][key_group_name].items():
                data['solution'][key_group_name+'_'+key] = v
            del data['solution'][key_group_name]
        
        gen_paths = glob(benchmark_path+'/'+n+'/*_population.json')
        gens = [json.load(open(p, 'r')) for p in gen_paths]
        data['population'] = []
        translator = data['translator']
        
        for gen in gens:
            for p in gen['population']:
                genes = p['solution']
                if type(genes) == list:
                    s = translator.decode(genes)
                else:
                    s = genes
                s['plm'] = s[n]
                del s[n]
                s['plm']['l1_dim'] = s['plm']['l1_dim']/plm_sizes[n]
                s['plm']['l2_dim'] = s['plm']['l2_dim']/plm_sizes[n]
                
                for key_group_name in ['plm', 'final']:
                    for key, v in s[key_group_name].items():
                        s[key_group_name+'_'+key] = v
                    del s[key_group_name]
                
                precision = p['metrics']['precision_score_w_05']
                roc = p['metrics']['ROC AUC W']
                
                data['population'].append({'params': s, 
                                           'precision': precision, 'roc': roc})
                #print(data['population'][-1])
        
        data['population'].sort(key = lambda p: p['roc'])
        data['population_best'] = data['population'][-32:]
        
    
    return solutions

def load_final_solutions(benchmark_path):
    val_jsons = glob(benchmark_path+'/*.json')
    vals = [json.load(open(p, 'r'))['validation'] for p in val_jsons]
    
    solution_jsons = [p.replace('.json', '/solution.json') for p in val_jsons]
    solutions = [json.load(open(p, 'r')) for p in solution_jsons]
    
    names = [path.basename(p).replace('.json', '') for p in val_jsons]
    
    solutions = {
        n: {'solution': s, 'metrics': v}
        for v, s, n in zip(vals, solutions, names)
    }
    
    for n, data in solutions.items():
        data['solution']['plm'] = data['solution'][n]
        del data['solution'][n]
        
        for key_group_name in ['plm', 'final']:
            for key, v in data['solution'][key_group_name].items():
                data['solution'][key_group_name+'_'+key] = v
            del  data['solution'][key_group_name]
    
    return solutions

def load_gens_df(benchmark_path):
    #print('model dirs:', benchmark_path+'/*')
    model_dirs = [d for d in glob(benchmark_path+'/*') if path.isdir(d)]
    
    rows = []
    for model_dir in model_dirs:
        n = path.basename(model_dir)
        translator_path = model_dir + '/params_dict_custom.json'
        if path.exists(translator_path) and n in plm_sizes:
            translator = ProblemTranslator(None, raw_values=json.load(open(translator_path, 'r')))
            gen_jsons = glob(model_dir+'/gen_*_population.json')
            for json_path in gen_jsons:
                gen_n = int(path.basename(json_path).split('_')[1])
                gen_pop = json.load(open(json_path, 'r'))
                for x in gen_pop['population']:
                    genes = x['solution']
                    if type(genes) == list:
                        s = translator.decode(genes)
                    else:
                        s = genes
                    s['plm'] = s[n]
                    del s[n]
                    s['plm']['l1_dim'] = s['plm']['l1_dim']/plm_sizes[n]
                    s['plm']['l2_dim'] = s['plm']['l2_dim']/plm_sizes[n]
                    
                    for key_group_name in ['plm', 'final']:
                        for key, v in s[key_group_name].items():
                            s[key_group_name+'_'+key] = v
                        del s[key_group_name]
                    
                    new_row = {'model': n, 'gen': gen_n}
                    for m_name, m_value in x['metrics'].items():
                        new_row[m_name] = m_value
                    for p_name, p_value in s.items():
                        new_row[p_name] = p_value
                    
                    rows.append(new_row)
                    #print(new_row)
    
    df = pd.DataFrame(rows)

    return df

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
        ax.plot(x, y, label=model_name, linewidth=5, alpha=0.7, color=model_colors[model_name])
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
        for m in ['f1_score_w_06', 'ROC AUC W', 'precision_score_w_06']:
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

def plot_final_solution_performance(benchmark_path):
    solutions = load_final_solutions(benchmark_path)

    precision = []
    roc = []
    names = []

    for name, v in solutions.items():
        precision.append(v['metrics']['f1_score_w_05'])
        roc.append(v['metrics']['ROC AUC W'])
        names.append(name)

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.scatter(precision, roc, s=160)
    for i, txt in enumerate(names):
        ax.annotate(txt.upper().replace('_', ' '), (precision[i], roc[i]), ha='center', va='bottom')
    ax.set_xlabel('F1 Weighted')
    ax.set_ylabel('ROC AUC Weighted')
    ax.set_title('M.F. Classification Performance of PLMs')
    fig.tight_layout()
    
    fig.savefig(benchmark_path+'/model_performance.png', dpi=120)
    
if __name__ == '__main__':
    benchmark_path = '/home/pita/experiments/base_benchmark_4'
    #benchmark_path = sys.argv[1]
    iterative_gens_draw(benchmark_path)
    
    