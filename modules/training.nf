
process make_trainer {
    publishDir "${params.optimization_dir}/${params.exp_name}/trainer", mode: 'copy'

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
    python -u ${params.src_dir}/main_train.py \
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
