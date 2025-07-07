import json
from os import path
from glob import glob
import sys
import matplotlib.pyplot as plt

def load_swarm_params_and_results_jsons(full_swarm_exp_dir):
    node_dirs = glob(full_swarm_exp_dir + '/Level-*')
    node_dicts = [x+'/exp_params.json' for x in node_dirs]
    node_results = [x+'/exp_results.json' for x in node_dirs]

    params_jsons = []
    results_jsons = []
    for exp_params, exp_results in zip(node_dicts, node_results):
        if not path.exists(exp_params):
            params_jsons.append(exp_params.replace('exp_params.json', 'standard_params.json'))
        else:
            params_jsons.append(exp_params)
        std_results = exp_results.replace('exp_results.json', 'standard_results.json')
        if path.exists(exp_results):
            results_jsons.append(exp_results)
        elif path.exists(std_results):
            results_jsons.append(std_results)
        else:
            results_jsons.append(None)

    return params_jsons, results_jsons

def draw_cv_relevance(full_swarm_exp_dir: str, output_dir: str):
    params_jsons, results_jsons = load_swarm_params_and_results_jsons(full_swarm_exp_dir)
    
    #n_proteins = []
    #node_names = []
    auprc_difs = []
    roc_auc_difs = []
    for exp_params, exp_results in zip(params_jsons, results_jsons):
        params = json.load(open(exp_params, 'r'))
        if exp_results is not None:
            results = json.load(open(exp_results, 'r'))
            auprc_w = results['validation']['AUPRC W']*100
            roc_auc_w = results['validation']['ROC AUC W']*100

            base_auprc_ws = [x['AUPRC W']*100 for x in results['base_model_validations']]
            base_roc_auc_ws = [x['ROC AUC W']*100 for x in results['base_model_validations']]

            difs = [auprc_w - x for x in base_auprc_ws]
            difs2 = [roc_auc_w - x for x in base_roc_auc_ws]

            auprc_difs.extend(difs)
            roc_auc_difs.extend(difs2)
            #auprc_difs.append(auprc_w - max(base_auprc_ws))
            #roc_auc_difs.append(roc_auc_w - max(base_roc_auc_ws))
    print(auprc_difs)
    print(roc_auc_difs)
    #Create box plot of AUPRC diferences and ROC AUC differences using matplotlib
    plt.figure(figsize=(5, 8))
    plt.boxplot([auprc_difs, roc_auc_difs], labels=['AUPRC Gains', 'ROC AUC Gains'])
    plt.title('Classification Performance Gains from Cross-Validation')
    plt.ylabel('Difference (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.join(output_dir, 'cv_relevance_boxplot.png'))
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python custom_statistics.py <full_swarm_exp_dir> <output_dir>")
        sys.exit(1)

    full_swarm_exp_dir = sys.argv[1]
    output_dir = sys.argv[2]

    draw_cv_relevance(full_swarm_exp_dir, output_dir)
    print(f"Statistics saved to {output_dir}/cv_relevance_boxplot.png")
    