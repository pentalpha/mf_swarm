import sys
import pandas as pd

from util_base import run_command

if __name__ == "__main__":
    dimension_db_releases_dir = sys.argv[1]
    #hint: 1
    dimension_db_release_n = sys.argv[2]
    datasets_dir = sys.argv[3]
    #hint: 30
    min_proteins_per_mf    = int(sys.argv[4])
    val_perc    = float(sys.argv[5])
    test_perc    = float(sys.argv[6])
    base_benchmark_tsv_path = sys.argv[7]
    pair_benchmark_dir    = sys.argv[8]
    output_script    = sys.argv[9]
    print(sys.argv)

    run_command(['mkdir -p', pair_benchmark_dir])

    performances_df = pd.read_csv(base_benchmark_tsv_path, sep='\t')

    feature_pairs = set()

    n_top = 2
    models_sorted = performances_df['model'].to_list()
    best_models = models_sorted[:n_top]
    for x in best_models:
        for y in best_models:
            if y != x:
                first, second = sorted([x, y])
                feature_pairs.add((first, second))
        if models_sorted[-1] != x:
            feature_pairs.add((x, models_sorted[-1]))
    print('Pairs to test:')

    cmds = ""
    outputs = []
    for a, b in feature_pairs:
        name = a+'-'+b
        local_dir = pair_benchmark_dir + '/' + name
        print(a, b)

        new_cmd = ("python src/model_pair_benchmark.py"
            +" "+dimension_db_releases_dir
            +" "+str(dimension_db_release_n)
            +" "+datasets_dir
            +" "+str(min_proteins_per_mf)
            +" "+str(val_perc)
            +" "+str(test_perc)
            +" "+base_benchmark_tsv_path
            +" "+local_dir)
        outputs.append(pair_benchmark_dir+'/'+name+'.json')
        cmds += new_cmd + '\n'

    final_json = pair_benchmark_dir + '/pair_results.json'
    cmds += ' '.join(["python", "src/merge_jsons.py"]+ outputs + [final_json])
    
    open(output_script, 'w').write(cmds)