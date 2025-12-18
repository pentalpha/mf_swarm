import gzip
from math import floor
from collections import defaultdict
import json
from os import path
from glob import glob

import numpy as np
import obonet
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches
from matplotlib.patches import Patch

from mf_swarm_lib.utils.parsing import load_swarm_params_and_results_jsons
from mf_swarm_lib.utils.parsing import (load_gens_df, load_final_solutions, 
    load_solutions, load_taxa_populations)

model_colors = {
    'ankh_base': 'red', 'ankh_large': 'darkred', 
    'esm2_t6': '#8FF259', 'esm2_t12': '#43D984', 
    'esm2_t30': '#3F1C34', 'esm2_t33': '#FFF955', 
    'esm2_t36': '#AD00B0', 
    'prottrans': 'blue'
}

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

def draw_swarm_panel(full_swarm_exp_dir: str, output_dir: str):
    node_dirs = glob(full_swarm_exp_dir + '/Level-*')
    node_dicts = [x+'/exp_params.json' for x in node_dirs]
    node_results = [x+'/exp_results.json' for x in node_dirs]
    
    n_proteins = []
    node_names = []
    auprc_ws = []
    roc_auc_ws = []
    test_fmax = []
    val_fmax = []
    node_levels = []
    node_pos_in_level = []
    n_labels = {}
    for exp_params, exp_results in zip(node_dicts, node_results):
        if not path.exists(exp_params):
            exp_params = exp_params.replace('exp_params.json', 'standard_params.json')
        params = json.load(open(exp_params, 'r'))

        node_names.append(params['node_name'])
        n_proteins.append(len(params['node']['id']))
        node_levels.append(int(node_names[-1].split('_')[0].split('-')[1]))
        n_labels[node_names[-1]] = len(params['node']['go'])

        std_results = exp_results.replace('exp_results.json', 'standard_results.json')

        if path.exists(exp_results):
            results = json.load(open(exp_results, 'r'))
            auprc_ws.append(results['validation']['AUPRC W'])
            roc_auc_ws.append(results['validation']['ROC AUC W'])
            val_fmax.append(results['validation']['Fmax'])
            test_fmax.append(results['test']['Fmax'])
        elif path.exists(std_results):
            results = json.load(open(std_results, 'r'))
            auprc_ws.append(results['validation']['AUPRC W'])
            roc_auc_ws.append(results['validation']['ROC AUC W'])
            val_fmax.append(results['validation']['Fmax'])
            test_fmax.append(results['test']['Fmax'])
        else:
            #print('No results at', exp_results)
            auprc_ws.append(None)
            roc_auc_ws.append(None)
            val_fmax.append(None)
            test_fmax.append(None)
    
    y_positions = []
    for i in range(len(node_levels)):
        level_names = [n for j, n in enumerate(node_names) if node_levels[j] == node_levels[i]]
        node_name = node_names[i]
        names_sorted = sorted(level_names, 
            key=lambda n: int(n.split('Freq-')[1].split('-')[0]), reverse=True)
        pos_in_level = names_sorted.index(node_name)
        node_pos_in_level.append(pos_in_level)
        if pos_in_level % 2 == 0:
            y_positions.append(node_levels[i]-0.125)
        else:
            y_positions.append(node_levels[i]+0.125)
    x_positions = []
    for index in range(len(y_positions)):
        x_positions.append(node_pos_in_level[index]*0.25)
    deepest_level = max(node_levels)
    levels_width = max(node_pos_in_level)
    auprc_min = min([x for x in auprc_ws if x is not None])
    auprc_min = round(auprc_min, 2)-0.05
    auprc_max = max([x for x in auprc_ws if x is not None])
    proteins_min = min(n_proteins)
    proteins_max = max(n_proteins)
    labels_min = min([x for x in n_labels.values()])
    labels_max = max([x for x in n_labels.values()])
    fmax_max = max(test_fmax + val_fmax)
    fmax_min = min(test_fmax + val_fmax)
    m1 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=auprc_min, vmax=auprc_max), 
        cmap=cm.seismic_r)
    m2 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=proteins_min, vmax=proteins_max), 
        cmap=cm.seismic)
    m3 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=labels_min, vmax=labels_max), 
        cmap=cm.seismic)
    m4 = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=fmax_min, vmax=fmax_max), 
        cmap=cm.seismic_r)
    plot_w = 5.5
    plot_h = 11
    fig1, ax1 = plt.subplots(1,1, figsize=(plot_w,plot_h))
    fig2, ax2 = plt.subplots(1,1, figsize=(plot_w,plot_h))
    fig3, ax3 = plt.subplots(1,1, figsize=(plot_w,plot_h))
    fig4, ax4 = plt.subplots(1,1, figsize=(plot_w,plot_h))
    fig5, ax5 = plt.subplots(1,1, figsize=(plot_w,plot_h))

    metrics_labels = ['AUPRC Weighted', 'Proteins for Train/test', 'GO IDs Predicted',
                      'Cross Validation Fmax', 'Validation Set Fmax']
    axes = [ax1, ax2, ax3, ax4, ax5]
    figs = [fig1, fig2, fig3, fig4, fig5]
    color_maps = [m1, m2, m3, m4, m4]

    for ax in axes:
        ax.scatter(x_positions, y_positions, c='white')

    for index in range(len(x_positions)):
        x_pos = x_positions[index]
        y_pos = y_positions[index]
        label = node_names[index]
        n_go_ids = n_labels[label]
        traintest_proteins = n_proteins[index]
        current_auprcw = auprc_ws[index]
        current_tfmax = test_fmax[index]
        current_vfmax = val_fmax[index]
        
        node_color1 = m1.to_rgba(current_auprcw) if current_auprcw is not None else 'white'
        node_color2 = m2.to_rgba(traintest_proteins)
        node_color3 = m3.to_rgba(n_go_ids)
        node_color4 = m4.to_rgba(current_tfmax)
        node_color5 = m4.to_rgba(current_vfmax)
        
        node_pos = (x_pos, y_pos)
        node_w = 0.25
        node_h = 0.5
        node_circle1 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color1, edgecolor='black')
        ax1.add_patch(node_circle1)
        node_circle2 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color2, edgecolor='black')
        ax2.add_patch(node_circle2)
        node_circle3 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color3, edgecolor='black')
        ax3.add_patch(node_circle3)
        node_circle4 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color4, edgecolor='black')
        ax4.add_patch(node_circle4)
        node_circle5 = patches.Ellipse(node_pos, node_w, node_h, 
            linewidth=2, facecolor=node_color5, edgecolor='black')
        ax5.add_patch(node_circle5)
    
    #ax.get_xaxis().set_visible(False)
    #ax.set_xscale('log', base=30)
    min_maxes = [(auprc_min, auprc_max), 
        (proteins_min, proteins_max), 
        (labels_min, labels_max),
        (fmax_min, fmax_max),
        (fmax_min, fmax_max)]
    for fig, ax, label, m, min_max in zip(figs, axes, metrics_labels, color_maps, min_maxes):
        ax.set_ylabel("Gene Ontology Level", fontsize=14)
        #ax.set_xlabel("Models in Layer", fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([1,2,3,4,5,6,7])
        ax.set_xticks([])
        #ax.xaxis.tick_top()
        #ax.xaxis.set_label_position('top')

        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(label+"\nof Base Models", fontsize=16)
        metric_min, metric_max = min_max
        fourth = (metric_max - metric_min) / 4
        point_2 = metric_min + fourth
        point_3 = metric_min + fourth *2
        point_4 = metric_max - fourth
        cbar = fig.colorbar(m, ax=ax, label=label, fraction=0.09,
            ticks=[metric_min, point_2, point_3, point_4, metric_max])
        cbar.ax.set_yticklabels(
            cbar.ax.get_yticklabels(), rotation=90, ha='left', va='center')
        #cbar.ax.yaxis.label.set_rotation(45)
        fig.tight_layout()
        #plt.show()
        label_norm = label.lower().replace(' ', '_').replace('/', '_')
        output_path = output_dir + '/swarm_' + label_norm + '.png'
        fig.savefig(output_path, dpi=120)

def plot_gens_evol(gens_df, output_path, metric_to_plot):
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    all_gens = set()
    for model_name, model_rows in gens_df.groupby('model'):
        x = []
        y = []
        for gen_n, model_gen_rows in model_rows.groupby('gen'):
            max_val = model_gen_rows[metric_to_plot].max()
            x.append(gen_n)
            y.append(max_val)
        all_gens.update(x)
        if not model_name in model_colors:
            names = model_name.split('-')
            if len(names) == 2:
                ax.plot(x, y, label=names[0], linewidth=6, alpha=0.8, color=model_colors[names[0]])
                ax.plot(x, y, label=names[1], linewidth=4, alpha=0.7, color=model_colors[names[1]])
        else:
            ax.plot(x, y, label=model_name, linewidth=6, alpha=0.7, color=model_colors[model_name])
        
    all_gens = [int(g) for g in sorted(all_gens)]
    ax.set_xticks(all_gens, [str(g) for g in all_gens])
    ax.set_xlim(min(all_gens), max(all_gens))
    ax.legend()
    ax.set_title('Metaheuristics Evolution - ' + metric_to_plot)
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=200)
    except Exception as err:
        pass

def iterative_gens_draw(benchmark_path, prev_n_gens=0):
    gens_df = load_gens_df(benchmark_path)
    if len(gens_df) > prev_n_gens:
        for m in ['f1_score_w_06', 'ROC AUC W', 'precision_score_w_06', 'fitness']:
            gens_plot_path = benchmark_path +'/evol-'+m+'.png'
            plot_gens_evol(gens_df, gens_plot_path, m)
        gens_df.to_csv(benchmark_path + '/all_gens.csv', sep=',')
    return len(gens_df)