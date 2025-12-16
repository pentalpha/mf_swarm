import sys
from tqdm import tqdm
from mf_swarm_lib.utils.go_expansion import gos_not_to_use, load_go_graph, expand_go_set

if __name__ == "__main__":
    id2gos_path = sys.argv[1]
    go_graph_path = sys.argv[2]

    not_use = gos_not_to_use()
    G = load_go_graph(go_graph_path)
    
    new_lines = []
    for rawline in tqdm(open(id2gos_path, 'r')):
        cells = [c for c in rawline.rstrip('\n').split('\t')]
        if len(cells) == 2:
            uniprot = cells[0]
            goids = cells[1].split(';')
            expanded_set = set()
            for goid in goids:
                expanded_set.update(expand_go_set(goid, G, not_use))
            expanded_set = sorted(expanded_set)
            new_lines.append(uniprot+'\t'+';'.join(expanded_set))

    with open(id2gos_path+'.expanded.tsv', 'w') as output:
        for rawline in tqdm(new_lines):
            output.write(rawline+'\n')
