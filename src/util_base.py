from glob import glob
import gzip
from os import path
import subprocess
from typing import List
import numpy as np

proj_dir = path.dirname(path.dirname(__file__))

#molecular function root
irrelevant_mfs = {'GO:0003674'}

plm_sizes = {
    'ankh_base': 768, 'ankh_large': 1536, 
    'esm2_t6': 320, 'esm2_t12': 480, 
    'esm2_t30': 640, 'esm2_t33': 1280, 
    'esm2_t36': 2560, 
    'prottrans': 1024,
    'taxa_profile_128': 128, 'taxa_profile_256': 256,
    'taxa_128': 128, 'taxa_256': 256
}

plm_billion_parameters = {
    'ankh_base': 0.650, 'ankh_large': 1.80, 
    'esm2_t6': 0.008, 'esm2_t12': 0.035, 
    'esm2_t30': 0.150, 'esm2_t33': 0.650, 
    'esm2_t36': 3, 
    'prottrans': 3,
    'taxa_profile_128': 0, 'taxa_profile_256': 0,
    'taxa_128': 0, 'taxa_256': 0
}

def calc_n_params(metaparameters_dict: dict, n_classes: int = 72):
    '''Calcula o número de parâmetros para um modelo, baseado em seus metaparametros'''
    
    input_models_parameters = []
    for key in plm_billion_parameters.keys():
        if key in metaparameters_dict:
            input_models_parameters.append(plm_billion_parameters[key]*1000000000)
    
    model_params = 0
    concat_len = 0
    for param in metaparameters_dict.keys():
        if param in plm_billion_parameters:
            l0 = plm_sizes[param]
            l1 = metaparameters_dict[param]["l1_dim"]
            l2 = metaparameters_dict[param]["l2_dim"]
            concat_len += l2

            module_params = l0 * l1 + l1*l2
            model_params += module_params
    
    l3 = metaparameters_dict['final']['final_dim']
    model_params += concat_len*l3 + l3*n_classes

    return model_params, input_models_parameters

def run_command(cmd_vec, stdin="", no_output=True):
    '''Executa um comando no shell e retorna a saída (stdout) dele.'''
    cmd_vec = " ".join(cmd_vec)
    #logging.info(cmd_vec)
    if no_output:
        #print(cmd_vec)
        result = subprocess.run(cmd_vec, shell=True)
        return result.returncode
    else:
        result = subprocess.run(cmd_vec, capture_output=True, 
            text=True, input=stdin, shell=True)
        return result.stdout

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def open_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'rt')
    else:
        return open(input_path, 'r')
    
def count_lines(input_path: str):
    stream = open_file(input_path)
    n = 0
    for line in stream:
        n += 1
    return n

def count_lines_large(input_path: str, chunk_size=500000):
    stream = open_file(input_path)
    temp_str = ""
    
    to_read = chunk_size
    for line in stream:
        temp_str += line

        to_read -= 1
        if not to_read:
            break

    temp_path = "file_chunk." + input_path.split('.')[-1]
    write_file(temp_path).write('\n'.join(temp_str))
    size_chunk = path.getsize(temp_path)
    size_full = path.getsize(input_path)

    chunks_in_full = size_full/size_chunk
    lines_f = chunks_in_full*chunk_size
    lines = int(lines_f)

    return lines
    
def write_file(input_path: str):
    if input_path.endswith('.gz'):
        return gzip.open(input_path, 'wt')
    else:
        return open(input_path, 'w')

def write_parsed_goa(annotations, file_path):
    output = write_file(file_path)
    for cells in annotations:
        line = '\t'.join(cells)
        output.write(line+'\n')

def concat_lists(vecs: list):
    vec = []
    for v in vecs:
        vec += v
    return vec

def concat_vecs(vecs: list):
    vec = []
    for v in vecs:
        vec += v
    return np.array(vec)

def concat_vecs_np(vecs: list):
    vec = []
    for v in vecs:
        vec += v.tolist()
    return np.array(vec)

def label_lists_to_onehot(label_lists: list):
    all_labels = set()
    for label_list in label_lists:
        all_labels.update(label_list)
    all_labels = sorted(list(all_labels))
    label_pos = {label: pos for pos, label in enumerate(all_labels)}

    n_labels = len(all_labels)
    print(n_labels, 'GO ids')
    one_hot = []
    for label_list in label_lists:
        vec = [0]*n_labels
        for go in label_list:
            vec[label_pos[go]] = 1
        one_hot.append(np.array(vec))

    return np.asarray(one_hot), all_labels

def create_go_labels(ids: List[str], ann: dict, labels_to_ignore: set = set()):
    label_lists = [ann[protid] for protid in ids]
    label_lists = [list(set(label_list) - labels_to_ignore) for label_list in label_lists]
    label_vecs, labels_sorted = label_lists_to_onehot(label_lists)
    return np.asarray(label_vecs), labels_sorted

def create_labels_matrix(labels: dict, ids_allowed: list, gos_allowed: list):
    label_vecs = []
    for protid in ids_allowed:
        gos_in_prot = labels[protid]
        one_hot_labels = [1 if go in gos_in_prot else 0 for go in gos_allowed]
        label_vecs.append(np.array(one_hot_labels))
    
    return np.asarray(label_vecs)

def get_items_at_indexes(all_items, indexes):
    new_items = []
    for i in indexes:
        new_items.append(all_items[i])
    return new_items


def create_params_for_features(features, bounds, convert_plm_dims=True):
    params_dict = {k: {k2: v2 for k2, v2 in v.items()} for k, v in bounds.items()}
    plm_base_params = params_dict['plm']
    for feature_name in features:
        if 'taxa' in feature_name:
            is_profile = 'profile' in feature_name
            if is_profile:
                params_dict[feature_name] = params_dict['taxa_profile']
            else:
                params_dict[feature_name] = params_dict['taxa']
        elif feature_name in plm_sizes:
            feature_len = plm_sizes[feature_name]
            if convert_plm_dims:
                params_dict[feature_name] = {
                    "l1_dim": [int(plm_base_params["l1_dim"][0]*feature_len), 
                                int(plm_base_params["l1_dim"][1]*feature_len)],
                    "l2_dim": [int(plm_base_params["l2_dim"][0]*feature_len), 
                                int(plm_base_params["l2_dim"][1]*feature_len)],
                    "dropout_rate": plm_base_params['dropout_rate'],
                    "leakyrelu_1_alpha": plm_base_params['leakyrelu_1_alpha']
                }
            else:
                params_dict[feature_name] = {
                    "l1_dim": [int(plm_base_params["l1_dim"][0]), 
                                int(plm_base_params["l1_dim"][1])],
                    "l2_dim": [int(plm_base_params["l2_dim"][0]), 
                                int(plm_base_params["l2_dim"][1])],
                    "dropout_rate": plm_base_params['dropout_rate'],
                    "leakyrelu_1_alpha": plm_base_params['leakyrelu_1_alpha']
            }
    
    del params_dict['taxa_profile']
    del params_dict['taxa']
    del params_dict['plm']

    return params_dict