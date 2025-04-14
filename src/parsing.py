from glob import glob
import json
from os import path
import pandas as pd
from tqdm import tqdm

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

def load_taxa_solutions(benchmark_path):
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
        for v, s, n, st in zip(vals, solutions, names, solution_translators) if n != 'None'
    }
    
    for n, data in solutions.items():
        if not n in data['solution']:
            print(n)
            print(data['solution'])
            raise Exception('Model not found in solution')
        
        taxa_feature_name1 = [k for k in data['solution'].keys() if 'taxa' in k][0]
        print(taxa_feature_name1)
        new_s = {}
        for key, v in data['solution'][taxa_feature_name1].items():
            new_s[taxa_feature_name1+'_'+key] = v
        data['solution'] = new_s
        
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
                taxa_feature_name = [k for k in s.keys() if 'taxa' in k][0]
                other_feature_name = [k for k in s.keys() if not ('taxa' in k)]
                for key, v in s[taxa_feature_name].items():
                    s[taxa_feature_name+'_'+key] = v
                del s[taxa_feature_name]
                for k in other_feature_name:
                    del s[k]
                
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

def load_final_metrics(benchmark_path):
    val_jsons = glob(benchmark_path+'/*.json')
    print('Loading', benchmark_path)
    solutions = {}
    for p in tqdm(val_jsons):
        data = json.load(open(p, 'r'))
        if 'validation' in data and 'test' in data and 'go_labels' in data:
            name = path.basename(p).replace('.json', '')
            params_dict_path = p.replace('.json', '/solution.json')
            params_dict = json.load(open(params_dict_path, 'r'))
            solutions[name] = {'solution': params_dict, 
                               'metrics': data['validation']}
    #vals = [json.load(open(p, 'r'))['validation'] for p in val_jsons]
    #names = [path.basename(p).replace('.json', '') for p in val_jsons]
    
    #solutions = {
    #    n: v
    #    for v, s, n in zip(vals, names)
    #}
    
    return solutions

def load_gens_df(benchmark_path):
    #print('model dirs:', benchmark_path+'/*')
    model_dirs = [d for d in glob(benchmark_path+'/*') if path.isdir(d)]
    
    rows = []
    for model_dir in model_dirs:
        n = path.basename(model_dir)
        translator_path = model_dir + '/params_dict_custom.json'
        if path.exists(translator_path):
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
                    if n in s and n in plm_sizes:
                        s['plm'] = s[n]
                        del s[n]
                        s['plm']['l1_dim'] = s['plm']['l1_dim']/plm_sizes[n]
                        s['plm']['l2_dim'] = s['plm']['l2_dim']/plm_sizes[n]
                        
                        for key_group_name in ['plm', 'final']:
                            for key, v in s[key_group_name].items():
                                s[key_group_name+'_'+key] = v
                            del s[key_group_name]
                    else:
                        for param_group in list(s.keys()):
                            param_dict = s[param_group]
                            if 'l1_dim' in param_dict and 'l2_dim' in param_dict:
                                param_dict['l1_dim'] = param_dict['l1_dim']/plm_sizes[param_group]
                                param_dict['l2_dim'] = param_dict['l2_dim']/plm_sizes[param_group]
                            
                            for key, v in param_dict.items():
                                s[param_group+'_'+key] = v
                            del s[param_group]
                    
                    new_row = {'model': n, 'gen': gen_n}
                    for m_name, m_value in x['metrics'].items():
                        new_row[m_name] = m_value
                    for p_name, p_value in s.items():
                        new_row[p_name] = p_value
                    
                    rows.append(new_row)
                    #print(new_row)
    
    df = pd.DataFrame(rows)

    return df