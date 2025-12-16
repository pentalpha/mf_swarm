from glob import glob
import json

def read_cluster_of_goids(swarm_dir):
    protein_ids = None
    params_jsons = glob(f"{swarm_dir}/Level-*/standard_params.json")
    cluster_n = {}
    params_jsons = [json.load(open(p, 'r')) for p in params_jsons]
    for p in params_jsons:
        p['Level'] = int(p['node_name'].split('-')[1].split('_')[0])
        p['MinFreq'] = int(p['node_name'].split('-')[2])
    params_jsons.sort(key=lambda p: (p['Level'], -p['MinFreq']))

    n = 0
    last_level = -1
    levels = {}
    for p in params_jsons:
        if protein_ids == None:
            protein_ids = set(p['node']['id'])
        goids = p['node']['go']
        level = p['Level']
        if level != last_level:
            n = 0
        else:
            n += 1

        for go_id in goids:
            levels[go_id] = level
            cluster_n[go_id] = n
        
        last_level = level
    
    return cluster_n, levels, protein_ids
