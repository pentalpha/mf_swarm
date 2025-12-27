nextflow.enable.dsl=2

/*
 * Default parameters
 */
params.exp_config = null
params.exp_name = null

// Generate timestamp for unique experiment naming
def timestamp = new Date().format("yyyy-MM-dd_HH-mm")
def experiment_dir = "${params.optimization_dir}/${params.exp_name}"
import groovy.json.JsonSlurper
//configs_used.json


// Modules
include { copy_configs } from './modules/utils'
include { create_dataset } from './modules/dataset'
include { make_trainer } from './modules/training'

workflow {
    if (!params.exp_config) {
        error "Please specify experimental config with --exp_config"
    }
    if (!params.exp_name) {
        error "Please specify experiment name with --exp_name"
    }
    if (!params.base_params_path) {
        error "Please specify base params path with --base_params_path"
        //params.base_params_path = "config/param_values/base_params_cafa6_v1.json"
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
