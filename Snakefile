from glob import glob
from os import path, mkdir
configfile: "config.yml"

base_benchmark_dir = path.expanduser(config['base_benchmark_dir'])
pairs_benchmark_dir = path.expanduser(config['pairs_benchmark_dir'])
dimension_db_dir = path.expanduser(config['dimension_db_dir'])
datasets_dir = path.expanduser(config['datasets_dir'])
release_n = config['release_n']
min_proteins_per_mf = config['min_proteins_per_mf']
val_perc = config['val_perc']
test_perc = config['test_perc']

real_test_perc = round(test_perc/(1.0-val_perc), 5)
print(val_perc, real_test_perc)

parquet_search_query = dimension_db_dir+'/release_'+str(release_n)+'/emb.*.parquet'
print(parquet_search_query)
parquets = glob(parquet_search_query)
print('Parquets:', parquets)

models_to_test = config["models_to_test"].split(',')
#models_to_test = ['esm2_t6', 'esm2_t12', 'esm2_t30', 
#    'ankh_base', 'prottrans', 'esm2_t33', 'ankh_large', 
#    'esm2_t36']
#models_to_test = ['esm2_t6', 'esm2_t12', 'ankh_base']

parquet_inputs = [dimension_db_dir+'/release_'+str(release_n)+'/emb.'+model_to_test+'.parquet'
    for model_to_test in models_to_test]
single_model_eval_outputs = [base_benchmark_dir+'/'+model_to_test+'.json' 
    for model_to_test in models_to_test]

for parquet_input, single_model_eval_output in zip(parquet_inputs, single_model_eval_outputs):
    rule:
        input:
            parquet_input
        output:
            single_model_eval_output
        shell:
            "conda run --live-stream -n mf_swarm_base"
                +" python src/base_benchmark.py "+dimension_db_dir
                +" "+str(release_n)
                +" "+datasets_dir
                +" "+str(min_proteins_per_mf)
                +" "+str(val_perc)
                +" "+str(real_test_perc)
                +" "+base_benchmark_dir
                +" {input}"

rule run_first_benchmark:
    input:
        'src/plotting.py',
        single_model_eval_outputs
    output:
        base_benchmark_dir + '/benchmark.tsv'
    shell:
        "conda run --live-stream -n plotting python src/summarize_base_benchmark.py " + base_benchmark_dir
        
rule run_pairs_benchmark:
    input:
        'src/model_pair_benchmark.py',
        base_benchmark_dir + '/benchmark.tsv'
    output:
        pairs_benchmark_dir + '/pair_results.json'
    shell:
        "conda run --live-stream -n mf_swarm_base"
            +" python src/model_pair_benchmark.py"
                +" "+dimension_db_dir
                +" "+str(release_n)
                +" "+datasets_dir
                +" "+str(min_proteins_per_mf)
                +" "+str(val_perc)
                +" "+str(real_test_perc)
                +" "+base_benchmark_dir + '/benchmark.tsv'
                +" "+pairs_benchmark_dir