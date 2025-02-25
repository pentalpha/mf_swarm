from glob import glob
import json
from os import path
import matplotlib.pyplot as plt
from decimal import Decimal
import sys

from param_translator import ProblemTranslator
from util_base import plm_sizes

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
                s = translator.decode(p['solution'])
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
    #benchmark_path = '/home/pita/experiments/base_benchmark_4'
    benchmark_path = sys.argv[1]
    plot_final_solution_performance(benchmark_path)
    plot_metrics(benchmark_path, final=False)