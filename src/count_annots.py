import sys
import gzip

from mf_swarm_lib.utils.plotting import annots_counts_plot

if __name__ == '__main__':
    mf_annot_path = sys.argv[1]
    #mf_annot_path = "/home/pita/data/protein_dimension_db/release_1/go.experimental.mf.tsv.gz"
    input_stream = gzip.open(mf_annot_path, 'rt')

    go_sets = {}
    for rawline in input_stream:
        protid, annots = rawline.rstrip('\n').split('\t')
        goids = annots.split(',')
        for goid in goids:
            if goid.startswith('GO:'):
                if goid not in go_sets:
                    go_sets[goid] = set()
                go_sets[goid].add(protid)

    go_ids_sorted = list(go_sets.keys())
    go_ids_sorted.sort(key = lambda g: len(go_sets[g]), reverse=True)
    prot_counts = [len(go_sets[g]) for g in go_ids_sorted]
    most_frequent = go_ids_sorted[0]
    if most_frequent == 'GO:0003674':
        prot_counts = prot_counts[:-1]
        go_ids_sorted = go_ids_sorted[:-1]

   annots_counts_plot(go_ids_sorted, prot_counts)    