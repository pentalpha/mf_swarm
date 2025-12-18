nextflow.enable.dsl=2

/*
 * Default parameters
 */
params.exp_config = null
params.exp_name = null

// Generate timestamp for unique experiment naming
//def timestamp = new Date().format("yyyy-MM-dd_HH-mm")
def experiment_dir = "${params.optimization_dir}/${params.exp_name}"
/*
 * Process to create dataset using dataset_maker.py
 */
import groovy.json.JsonSlurper
//configs_used.json

//Copy configs dict to experiment_dir/configs_used.json process
process copy_configs {
    publishDir "${experiment_dir}/configs_used.json", mode: 'copy'

    input:
    path configs_file
    output:
    path "configs_used.json", emit: configs_json
    script:
    """
    cp $configs_file configs_used.json
    """
}

process create_dataset {
    publishDir "${experiment_dir}", mode: 'copy'

    input:
    val split_config
    path dimension_db_dir
    val release_n

    output:
    path "dataset", emit: dataset_dir
    path "dataset/params.json", emit: dataset_params
    path "dataset/labels*.parquet", emit: dataset_labels
    path "dataset/features*.parquet", emit: dataset_features
    path "dataset/ids.txt", emit: dataset_ids
    path "dataset/go_ids.txt", emit: dataset_go_ids
    path "dataset/go_clusters.json.gz", emit: dataset_go_clusters

    script:
    """
    ls -sh $dimension_db_dir
    ls -sh $dimension_db_dir/$release_n
    python /src/dataset_maker.py \\
        -d $dimension_db_dir \\
        -r $release_n \\
        -o dataset \\
        -m ${split_config.min_proteins_per_mf} \\
        -v ${split_config.val_perc} \\
        -t ${split_config.dataset_type}
    """
}

process make_trainer {
    publishDir "${experiment_dir}/trainer", mode: 'copy'

    input:
    val experiment_name
    path dimension_db_dir
    val release_n
    path dataset_dir
    val min_proteins_per_mf
    val val_perc
    path base_params_path
    val n_jobs

    output:
    path "${experiment_name}", emit: experiment_trainer_dir

    script:
    """
    python /src/main_train.py \
        --experiment-name ${experiment_name} \
        --dimension-db-releases-dir ${dimension_db_dir} \
        --dimension-db-release-n ${release_n} \
        --dataset-dir ${dataset_dir} \
        --min-proteins-per-mf ${min_proteins_per_mf} \
        --val-perc ${val_perc} \
        --base-params-path ${base_params_path} \
        --n-jobs ${n_jobs}
    """
    
}

workflow {
    if (!params.exp_config) {
        error "Please specify experimental config with --exp_config"
    }
    if (!params.exp_name) {
        error "Please specify experiment name with --exp_name"
    }

    // Parse JSON config
    def jsonFile = file(params.exp_config)
    def paramsFile = file(params.base_params_path)
    def jsonMap = new JsonSlurper().parseText(jsonFile.text)
    
    // Set default dataset_type if missing
    if (!jsonMap.containsKey('dataset_type')) {
        jsonMap.dataset_type = 'full_swarm'
    }

    //Print experiment_dir, min proteins and other parameters:
    println "Experiment directory: ${experiment_dir}"
    println "Min proteins per MF: ${jsonMap.min_proteins_per_mf}"
    println "Val perc: ${jsonMap.val_perc}"
    println "Dataset type: ${jsonMap.dataset_type}"
    copy_configs(jsonFile)
    
    // Verify required fields
    if (!jsonMap.containsKey('min_proteins_per_mf') || !jsonMap.containsKey('val_perc')) {
        error "JSON config must contain 'min_proteins_per_mf' and 'val_perc'"
    }

    create_dataset(jsonMap, params.dimension_db_dir, params.release_n)

    make_trainer(params.exp_name, params.dimension_db_dir, params.release_n, 
        create_dataset.out.dataset_dir, jsonMap.min_proteins_per_mf, jsonMap.val_perc, 
        paramsFile, jsonMap.n_jobs)
}
