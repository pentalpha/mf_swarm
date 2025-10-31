import gzip
from math import floor
from collections import defaultdict
import requests
import io
import json
import sys
from glob import glob

import numpy as np
import obonet
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

swarm_dir = sys.argv[1]
pddb_dir = sys.argv[2]

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

print('Reading swarm')
cluster_n, distances_original, protein_ids = read_cluster_of_goids(swarm_dir)
go_obo_path = f'{pddb_dir}/go-basic.obo'

print('Reading annotations')
mf_annot_path = f'{pddb_dir}/go.experimental.mf.tsv.gz'
input_stream = gzip.open(mf_annot_path, 'rt')
go_freqs = {}
for rawline in input_stream:
    protid, annots = rawline.rstrip('\n').split('\t')
    if protid in protein_ids:
        goids = annots.split(',')
        for goid in goids:
            if goid.startswith('GO:'):
                if goid not in go_freqs:
                    go_freqs[goid] = 1
                else:
                    go_freqs[goid] += 1

# Download and parse GO ontology
print("reading Gene Ontology...")
url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
response = requests.get(url)
go_graph = obonet.read_obo(open(go_obo_path))
# Identify root node (molecular_function)
root = "GO:0003674"

# Filter for Molecular Function terms
mf_terms = []
for node, data in go_graph.nodes(data=True):
    if data.get('namespace') == 'molecular_function':
        mf_terms.append(node)
print('Mf terms:', len(mf_terms))
if root not in mf_terms:
    mf_terms.append(root)
# Create subgraph of Molecular Function terms
mf_graph = go_graph.subgraph(mf_terms).copy()
print(len(mf_graph.nodes()), "Molecular Function terms in the graph")
root_predecessors = list(mf_graph.predecessors(root))
print("Root predecessors:", root_predecessors)
# Inverter o grafo para percorrer do root para os filhos
print("Reversed graph created")
print("Number of edges in the graph:", len(mf_graph.edges()))
print("Number of edges in the graph:", len(mf_graph.nodes()))


# Calcular distâncias normalmente
'''distances = {}
for node in mf_graph.nodes():
    if node == root:
        distances[node] = 0
    else:
        paths = nx.all_simple_paths(mf_graph, source=node, target=root)
        lens = [len(path) for path in paths]
        longest_path = max(lens) if lens else 0
        shortest_path = min(lens) if lens else 0
        distances[node] = shortest_path + 1'''

distances = {}
print('Finding paths from MF terms to MF Root')
max_level = 11
for goid in mf_graph.nodes():
#for goid in tqdm(valid_goids):
    if goid != root:
        simple_paths = nx.all_simple_paths(go_graph, source=goid, target=root)
        simple_path_lens = [len(p) for p in simple_paths]
        try:
            mean_dist = floor(np.mean(simple_path_lens)-1)
            distances[goid] = min(max_level, mean_dist)
        except ValueError as err:
            print(simple_path_lens)
            print('No path from', goid, 'to', root)
            print(err)
            raise(err)
    else:
        distances[goid] = -1

print(len(distances))
print("Longest path distance from root:", max(distances.values()))
print("Shortest path distance from root:", min(distances.values()))
# Remove nodes that are not connected to the root
connected_nodes = set(distances.keys())
mf_graph = mf_graph.subgraph(connected_nodes).copy()

# Create hierarchical layout
level_groups = defaultdict(list)
for node, dist in distances.items():
    level_groups[dist].append(node) 

# Generate positions
pos = {}
max_nodes = max(len(nodes) for nodes in level_groups.values())
level_spacing = 0.2
for level, nodes in level_groups.items():
    for n in nodes:
        if not n in go_freqs:
            go_freqs[n] = 0
    nodes_sorted = sorted(nodes, key=lambda node: go_freqs[node], reverse=True)  # Sort for consistent ordering
    for i, node in enumerate(nodes_sorted):
        # Distribute nodes evenly horizontally
        x = (i + 0.5) / len(nodes) * max_nodes
        pos[node] = (x, level)  # Negative for top-down layout
#%%
pos[root] = (pos[root][0], -1)

min_x = min([p[0] for p in pos.values()])
max_x = max([p[0] for p in pos.values()])

#cluster_n[5] = []
for node in mf_graph.nodes():
    if not node in cluster_n:
        cluster_n[node] = 5

cluster_to_color = {
    0: "#BEF852",
    1: "#C164FF",
    2: '#FF932F',
    3: "#6985FF",
    5: "#CCCCCC"
}

# Create plot
fig, ax = plt.subplots(figsize=(10, 9))
print('List network')
node_labels = {node: data.get('name', node) 
               for node, data in mf_graph.nodes(data=True) 
               if node in distances}
#node_colors = [distances[node] for node in mf_graph.nodes()]

print('Draw network edges')
nx.draw_networkx_edges(
    mf_graph,
    pos,
    alpha=0.1,
    edge_color='gray',
    arrowsize=5,
    ax=ax
)

nodes_main_sort = sorted([node for node in mf_graph.nodes()], key=lambda node: cluster_n[node], reverse=True)
node_colors = [cluster_to_color[cluster_n[node]] for node in nodes_main_sort]
if not root in nodes_main_sort:
    nodes_main_sort = [root] + nodes_main_sort
    node_colors = ['red'] + node_colors
print('Draw network nodes')
nodes = nx.draw_networkx_nodes(
    mf_graph,
    pos,
    node_color=node_colors,
    nodelist = nodes_main_sort,
    #cmap=plt.cm.viridis,
    node_size=50,
    alpha=0.8,
    ax=ax,
    node_shape='s',
    edgecolors=None
)

print('Add labels only for root and level 1 nodes for readability')
label_nodes = [node for node in mf_graph.nodes() 
               if distances.get(node, 1000) <= 1 and cluster_n[node] < 4]

def short_label(l, max_len=25):
    if len(l) > max_len:
        return l[:max_len-3] + '...'
    else:
        return l

labels = nx.draw_networkx_labels(
    mf_graph,
    pos,
    labels={n: short_label(node_labels[n]) for n in label_nodes},
    font_size=9,
    ax=ax
)
for key,t in labels.items():
    t.set_va('top')
    t.set_ha('left')
    t.set_rotation(-40)

ax.text(pos[root][0], pos[root][1], 'Molecular Function', fontsize=12, ha='center',va='top')

'''print('Create colorbar')
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis,
    norm=plt.Normalize(vmin=min(distances.values()), vmax=max(distances.values())))
sm.set_array([])'''
#cbar = plt.colorbar(sm, orientation='horizontal', pad=0.05, ax=ax)
#cbar.set_label('Distance from Root', fontsize=12)

print('Final rendering plot')
ax.set_title(f"Hierarchy of Molecular Function Classifiers", fontsize=14)
ax.axis('on')
ax.set_ylabel('Mean Distance to Root')
ax.get_yaxis().set_ticks([1,2,3,4,5,6,7,8,9,10,11])
ax.set_xlabel('Molecular Functions (sorted by annotation frequency)')
ax.get_xaxis().set_ticks([])
# Enable the axis ticks and labels
ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)
ax.get_xaxis().set_ticks([])
ax.spines[['right', 'top', 'bottom']].set_visible(False)

from matplotlib.patches import Patch

width_base = max_x-min_x
x_padding = width_base*0.03
y_padding=0.2
y_start = 7-y_padding
height = y_padding+4+y_padding

ax.add_patch(plt.Rectangle(xy=(min_x-x_padding, y_start), width=width_base+x_padding*2, height=height, 
    edgecolor="#686868", fill=False, ls='--'))

ax.text(pos[root][0], 11.25, 'Last Levels Collapsed Into One', fontsize=9, ha='center',va='bottom')

legend_elements = [Patch(facecolor=cluster_to_color[0], edgecolor=cluster_to_color[0],
                    label='First Classifier of Level'),
                Patch(facecolor=cluster_to_color[1], edgecolor=cluster_to_color[1],
                    label='Second Classifier of Level'),
                Patch(facecolor=cluster_to_color[2], edgecolor=cluster_to_color[2],
                    label='Third Classifier of Level'),
                Patch(facecolor=cluster_to_color[3], edgecolor=cluster_to_color[3],
                    label='Fourth Classifier of Level'),
                Patch(facecolor=cluster_to_color[5], edgecolor=cluster_to_color[5],
                    label='Not classified')]
# Create the figure
ax.legend(handles=legend_elements, loc='lower right')

fig.tight_layout()
fig.savefig(f'{swarm_dir}/plots/go_molecular_function_hierarchy.png', dpi=200)