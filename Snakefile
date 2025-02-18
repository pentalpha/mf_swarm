from glob import glob
from os import path, mkdir
configfile: "config.yml"

dimension_db_dir = path.expanduser(config['dimension_db_dir'])
datasets_dir = path.expanduser(config['datasets_dir'])
release_n = config['release_n']
min_proteins_per_mf = config['min_proteins_per_mf']
val_perc = config['val_perc']

parquet_search_query = dimension_db_dir+'/release_'+str(release_n)+'/emb.*.parquet'
print(parquet_search_query)
parquets = glob(parquet_search_query)
print('Parquets:', parquets)
rule run_first_benchmark:
    input:
        parquets + ["src/base_benchmark.py"]
    shell:
        "conda run --live-stream -n mf_swarm_base"
            " python src/base_benchmark.py "+dimension_db_dir
            +" "+str(release_n)
            +" "+datasets_dir
            +" "+str(min_proteins_per_mf)
            +" "+str(val_perc)
        