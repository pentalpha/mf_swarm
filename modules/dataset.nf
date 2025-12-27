
process create_dataset {
    publishDir "${params.optimization_dir}/${params.exp_name}", mode: 'copy'

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
    python ${params.src_dir}/dataset_maker.py \\
        -d $dimension_db_dir \\
        -r $release_n \\
        -o dataset \\
        -m ${split_config.min_proteins_per_mf} \\
        -v ${split_config.val_perc} \\
        -t ${split_config.dataset_type}
    """
}
