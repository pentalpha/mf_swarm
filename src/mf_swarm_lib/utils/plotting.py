import gzip
from math import floor
from collections import defaultdict
import requests
import json

import numpy as np
import obonet
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mf_swarm_lib.utils.parsing import load_swarm_params_and_results_jsons

def plot_hierarchy(swarm_dir, pddb_dir, cluster_n, protein_ids):
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
    # response = requests.get(url) # Not used in original code
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
    
    # Inverter o grafo para percorrer do root para os filhos
    print("Number of edges in the graph:", len(mf_graph.edges()))
    print("Number of nodes in the graph:", len(mf_graph.nodes()))


    distances = {}
    print('Finding paths from MF terms to MF Root')
    max_level = 11
    for goid in mf_graph.nodes():
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
    
    for level, nodes in level_groups.items():
        for n in nodes:
            if not n in go_freqs:
                go_freqs[n] = 0
        nodes_sorted = sorted(nodes, key=lambda node: go_freqs[node], reverse=True)  # Sort for consistent ordering
        for i, node in enumerate(nodes_sorted):
            # Distribute nodes evenly horizontally
            x = (i + 0.5) / len(nodes) * max_nodes
            pos[node] = (x, level)  # Negative for top-down layout
    
    pos[root] = (pos[root][0], -1)

    min_x = min([p[0] for p in pos.values()])
    max_x = max([p[0] for p in pos.values()])

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
    nx.draw_networkx_nodes(
        mf_graph,
        pos,
        node_color=node_colors,
        nodelist = nodes_main_sort,
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

    print('Final rendering plot')
    ax.set_title(f"Hierarchy of Molecular Function Classifiers", fontsize=14)
    ax.axis('on')
    ax.set_ylabel('Mean Distance to Root')
    ax.get_yaxis().set_ticks([1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xlabel('Molecular Functions (sorted by annotation frequency)')
    ax.get_xaxis().set_ticks([])
    # Enable the axis ticks and labels
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)

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



def annots_counts_plot(go_ids_sorted, prot_counts):

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

def draw_cv_relevance(full_swarm_exp_dir: str, output_dir: str):
    params_jsons, results_jsons = load_swarm_params_and_results_jsons(full_swarm_exp_dir)
    
    #n_proteins = []
    #node_names = []
    auprc_difs = []
    roc_auc_difs = []
    for exp_params, exp_results in zip(params_jsons, results_jsons):
        params = json.load(open(exp_params, 'r'))
        if exp_results is not None:
            results = json.load(open(exp_results, 'r'))
            auprc_w = results['validation']['AUPRC W']*100
            roc_auc_w = results['validation']['ROC AUC W']*100

            base_auprc_ws = [x['AUPRC W']*100 for x in results['base_model_validations']]
            base_roc_auc_ws = [x['ROC AUC W']*100 for x in results['base_model_validations']]

            difs = [auprc_w - x for x in base_auprc_ws]
            difs2 = [roc_auc_w - x for x in base_roc_auc_ws]

            auprc_difs.extend(difs)
            roc_auc_difs.extend(difs2)
            #auprc_difs.append(auprc_w - max(base_auprc_ws))
            #roc_auc_difs.append(roc_auc_w - max(base_roc_auc_ws))
    print(auprc_difs)
    print(roc_auc_difs)
    #Create box plot of AUPRC diferences and ROC AUC differences using matplotlib
    plt.figure(figsize=(5, 8))
    plt.boxplot([auprc_difs, roc_auc_difs], labels=['AUPRC Gains', 'ROC AUC Gains'])
    plt.title('Classification Performance Gains from Cross-Validation')
    plt.ylabel('Difference (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.join(output_dir, 'cv_relevance_boxplot.png'))
    plt.close()