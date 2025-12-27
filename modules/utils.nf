
process copy_configs {
    publishDir "${params.optimization_dir}/${params.exp_name}", mode: 'copy', pattern: "configs_used.json"

    input:
    path configs_file

    output:
    path "configs_used.json", emit: configs_json

    script:
    """
    cp $configs_file configs_used.json
    """
}
