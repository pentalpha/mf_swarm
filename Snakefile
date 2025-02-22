from glob import glob
from os import path, mkdir
configfile: "config.yml"

experiments_dir = path.expanduser(config['experiments_dir'])
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

model_to_test = 'esm2_t6'

rule run_first_benchmark:
    input:
        parquets + ["src/base_benchmark.py"]
    output:
        experiments_dir+'/'+model_to_test+'.json'
    shell:
        "rm -f src/__pycache__/*"
        " && conda run --live-stream -n mf_swarm_base"
            " python src/base_benchmark.py "+dimension_db_dir
            +" "+str(release_n)
            +" "+datasets_dir
            +" "+str(min_proteins_per_mf)
            +" "+str(val_perc)
            +" "+str(real_test_perc)
            +" "+experiments_dir
            +" "+model_to_test
        