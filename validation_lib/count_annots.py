import sys
import gzip

#mf_annot_path = sys.argv[1]
mf_annot_path = "/home/pita/data/protein_dimension_db/release_1/go.experimental.mf.tsv.gz"
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


#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8,4))
x = [n for n, _ in enumerate(go_ids_sorted)]
ax.set_axisbelow(True)
ax.grid(visible=True, which='major', axis='y', linestyle='dashed')
ax.plot(x, prot_counts, color='#66FF66')
ax.fill_between(x, prot_counts, color='#66FF66')
ax.set_yscale('log')
ax.set_ylabel('Número de Proteínas Anotadas')
ax.set_xlabel('Funções Moleculares do Gene Ontology\n(da mais frequente à menos frequente)')
#ax.get_xaxis().set_visible(False)
ax.get_xaxis().set_ticks([])
from matplotlib.ticker import ScalarFormatter
ax.yaxis.set_major_formatter(ScalarFormatter())
fig.tight_layout()
fig.savefig('img/mf_annots_counts.png', dpi=300)