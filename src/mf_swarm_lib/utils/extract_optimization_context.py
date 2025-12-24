import json
from glob import glob
from os import path
import sys
from copy import deepcopy

def node_history_str(node_dir, max_lines=8):
    meta_params_paths = glob(f'{node_dir}/standard_model/fold_*/metaparams.json')
    meta_params = [json.load(open(p, 'r')) for p in meta_params_paths]
    hist_by_fold = {}
    for json_path, meta_params_dict in zip(meta_params_paths, meta_params):
        fold = path.basename(path.dirname(json_path)).replace('fold_', '')
        history = meta_params_dict['history']
        metric_names = sorted(history[0].keys())
        header = ', '.join(metric_names)
        lines = []
        for h in history:
            lines.append(', '.join([str(h[n]) for n in metric_names]))
        
        if len(lines) > max_lines:
            #get first, last and (max_lines - 2) lines uniformely from the middle
            #making sure to there is an uniform distance between the sampled lines
            lines = [lines[0]] + lines[1:-1: (len(lines) - 2) // (max_lines - 2)] + [lines[-1]]
        
        lines = [header] + lines
        hist_by_fold[fold] = lines
    
    return hist_by_fold

def extract_info_from_swarm(swarm_dir):
    metrics_csv = f'{swarm_dir}/raw_metrics_by_label.json'
    metrics_dict = json.load(open(metrics_csv, 'r'))
    
    full_context = {
        'final_validation_results': {
            'fmax': metrics_dict['Fmax'],
            'auprc': metrics_dict['AUPRC W'],
            'precision': metrics_dict['boolean']['Precision W'],
            'recall': metrics_dict['boolean']['Recall W'],
            'f1': metrics_dict['boolean']['F1 W']
        },
        'training_history': {},
        'test_results': {}
    }

    node_dirs = glob(f'{swarm_dir}/Level-*_Freq-*')
    node_dirs = [d for d in node_dirs if path.isdir(d)]
    for node_dir in node_dirs:
        hist_dict = node_history_str(node_dir, max_lines=10)
        full_context['training_history'][path.basename(node_dir)] = hist_dict
        standard_results_path = f'{node_dir}/standard_results.json'
        standard_results = json.load(open(standard_results_path, 'r'))
        test_results = standard_results['test']
        metrics = ['AUPRC W', 'Fmax', 'ROC AUC W']
        test_metrics = {m: test_results[m] for m in metrics}
        full_context['test_results'][path.basename(node_dir)] = test_metrics
    return full_context

def medium_description(full_context):
    return {
        'final_validation_results': full_context['final_validation_results'],
        'test_results': full_context['test_results']
    }

def short_description(full_context):
    return {
        'final_validation_results': full_context['final_validation_results'],
    }

def sort_experiments(experiments):
    #{'metaparameters': {...}, 'result': {...}}
    experiments.sort(
        key=lambda x: x['result']['final_validation_results']['fmax'],
        reverse=False)

def optimization_report(experiments, dataset_config, bounds_dict, commentary = ''):
    sort_experiments(experiments)

    worst = experiments[0]
    middle = experiments[len(experiments)//2]
    good = experiments[-3]
    very_good = experiments[-2]
    best = experiments[-1]

    worst['result'] = short_description(worst['result'])
    middle['result'] = short_description(middle['result'])
    #print(good['result'], file=sys.stderr)
    good['result'] = medium_description(good['result'])
    very_good['result'] = medium_description(very_good['result'])

    optimization_report_json = {
        'input_bounds': bounds_dict,
        'dataset_config': dataset_config,
        'experiments': {
            'worst': worst,
            'middle': middle,
            'good': good,
            'very_good': very_good,
            'best': best
        },
        'best_params': best['metaparameters'],
        'best_result': short_description(best['result']),
        'commentary': commentary
    }

    return optimization_report_json

def short_optimization_report(full_report) -> dict:
    return {
        'input_bounds': full_report['input_bounds'],
        'dataset_config': full_report['dataset_config'],
        'best_params': full_report['best_params'],
        'best_result': full_report['best_result']
    }

def medium_optimization_report(full_report) -> dict:
    return {
        'input_bounds': full_report['input_bounds'],
        'dataset_config': full_report['dataset_config'],
        'experiments': {
            'best': full_report['experiments']['best']
        },
        'best_params': full_report['best_params'],
        'best_result': full_report['best_result'],
        'commentary': full_report['commentary']
    }

def calc_tokens_simple(full_str) -> int:
    #approximate the number of llm tokens in the json string
    s = full_str.replace(',', ' ').replace('{', ' ').replace('}', ' ')
    s = full_str.replace('\n', ' ').replace('  ', ' ').replace('   ', ' ')
    n = len(s.split())
    return n

def compress_to_max_tokens(optimization_reports, max_tokens = 5000):
    print(json.dumps(optimization_reports[-1], indent=4, ensure_ascii=False), file = sys.stderr)
    total_tokens = sum([calc_tokens_simple(json.dumps(c)) 
        for c in optimization_reports])
    print(total_tokens, 'tokens', file = sys.stderr)
    if total_tokens > max_tokens:
        #Compress reports (except for the last one)
        #First with medium compression. If needed, start using short compression
        compression_func = medium_optimization_report
        next_to_compress = 0
        print('Doing medium compression', file = sys.stderr)
        while total_tokens > max_tokens:
            if next_to_compress == (len(optimization_reports) - 1):
                if compression_func == medium_optimization_report:
                    print(total_tokens, 'tokens', file = sys.stderr)
                    next_to_compress = 0
                    compression_func = short_optimization_report
                    print('Doing short compression', file = sys.stderr)
                else:
                    break
            optimization_reports[next_to_compress] = compression_func(optimization_reports[next_to_compress])
            total_tokens = sum([calc_tokens_simple(json.dumps(c)) 
                for c in optimization_reports])
            next_to_compress += 1
        print(total_tokens, 'tokens', file = sys.stderr)

        before = len(optimization_reports)
        if total_tokens > max_tokens:
            print('Deleting oldest reports', file = sys.stderr)
            while total_tokens > max_tokens:
                #delete first line, until total_tokens < max_tokens
                optimization_reports.pop(0)
                total_tokens = sum([calc_tokens_simple(json.dumps(c)) 
                    for c in optimization_reports])
            after = len(optimization_reports)
            print(before - after, 'reports deleted', file = sys.stderr)
        print(total_tokens, 'tokens', file = sys.stderr)
    
    return optimization_reports

if __name__ == '__main__':
    #Extract info from local swarm directory
    context_full = extract_info_from_swarm('./')
    metaparameters_dict = json.load(open('./standard_params.json', 'r'))
    del metaparameters_dict['input_dims']
    experiment_example = {
        'metaparameters': metaparameters_dict,
        'result': context_full
    }
    print(experiment_example, file=sys.stderr)
    param_bounds_example = {
        'param_group_a': {
            'param1': [0, 10],
            'param2': [0, 1]
        }
    }
    dataset_config_example = {
        'dataset_type': 'small_swarm',
        'validation_size': 0.2,
        'test_size': 0.2
    }
    #create 20 deep copies of the experiment_example dictionary
    exp_list = [deepcopy(experiment_example) for _ in range(20)]
    report_example = optimization_report(
        commentary = 'This is why I did this experiment: blablabla',
        experiments = exp_list,
        bounds_dict = param_bounds_example,
        dataset_config = dataset_config_example
    )
    reports_list = [deepcopy(report_example) for _ in range(25)]
    compressed_report = compress_to_max_tokens(reports_list, 1600)
    print(json.dumps(compressed_report, indent=4, ensure_ascii=False))
