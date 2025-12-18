nextflow.enable.dsl=2

/*
 * Default parameters
 */
params.exp_config = null
params.exp_name = null

// Generate timestamp for unique experiment naming
def timestamp = new Date().format("yyyy-MM-dd_HH-mm")

/*
 * Process to create dataset using dataset_maker.py
 */
import groovy.json.JsonSlurper

process create_dataset {
    publishDir "${params.optimization_dir}/${params.exp_name}_${timestamp}/dataset", mode: 'copy'

    input:
    val split_config
    val timestamp
    path dimension_db_dir
    val release_n
    val dataset_internal_name

    output:
    path "${dataset_internal_name}", emit: dataset_dir

    script:
    """
    echo "Parameters:"
    echo "  Min Proteins: ${split_config.min_proteins_per_mf}"
    echo "  Val Perc: ${split_config.val_perc}"
    echo "  Type: ${split_config.dataset_type}"
    echo "  Output: ${dataset_internal_name}"
    
    # Run dataset_maker.py
    # We pass the values directly from the grooy map
    ls -sh $dimension_db_dir
    ls -sh $dimension_db_dir/$release_n
    python /src/dataset_maker.py \\
        -d $dimension_db_dir \\
        -r $release_n \\
        -o $dataset_internal_name \\
        -m ${split_config.min_proteins_per_mf} \\
        -v ${split_config.val_perc} \\
        -t ${split_config.dataset_type}
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
    def jsonMap = new JsonSlurper().parseText(jsonFile.text)
    
    // Set default dataset_type if missing
    if (!jsonMap.containsKey('dataset_type')) {
        jsonMap.dataset_type = 'full_swarm'
    }

    def dataset_internal_name = jsonMap.dataset_type + "_" + timestamp
    
    // Verify required fields
    if (!jsonMap.containsKey('min_proteins_per_mf') || !jsonMap.containsKey('val_perc')) {
        error "JSON config must contain 'min_proteins_per_mf' and 'val_perc'"
    }

    create_dataset(jsonMap, timestamp, params.dimension_db_dir, params.release_n, dataset_internal_name)
}
