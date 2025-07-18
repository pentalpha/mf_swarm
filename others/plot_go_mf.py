import obonet
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import requests
import io

# Download and parse GO ontology
print("Downloading Gene Ontology...")
url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
response = requests.get(url)
go_graph = obonet.read_obo(io.StringIO(response.text))
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
distances = {}
'''for node in nx.dfs_postorder_nodes(mf_graph, source=root):
    if node == root:
        distances[node] = 0
        continue
    parents = [p for p in mf_graph.predecessors(node) 
               if p in mf_graph and p in distances]
    if parents:
        distances[node] = max(distances[p] for p in parents) + 1'''
for node in mf_graph.nodes():
    if node == root:
        distances[node] = 0
    else:
        paths = nx.all_simple_paths(mf_graph, source=node, target=root)
        lens = [len(path) for path in paths]
        longest_path = max(lens) if lens else 0
        shortest_path = min(lens) if lens else 0
        distances[node] = shortest_path + 1
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
    nodes_sorted = sorted(nodes)  # Sort for consistent ordering
    for i, node in enumerate(nodes_sorted):
        # Distribute nodes evenly horizontally
        x = (i + 0.5) / len(nodes) * max_nodes
        pos[node] = (x, -level)  # Negative for top-down layout

# Create plot
fig, ax = plt.subplots(figsize=(14, 18))
node_labels = {node: data.get('name', node) 
               for node, data in mf_graph.nodes(data=True) 
               if node in distances}
node_colors = [distances[node] for node in mf_graph.nodes()]

# Draw network
nx.draw_networkx_edges(
    mf_graph,
    pos,
    alpha=0.3,
    edge_color='gray',
    arrowsize=5,
    ax=ax
)
nodes = nx.draw_networkx_nodes(
    mf_graph,
    pos,
    node_color=node_colors,
    cmap=plt.cm.viridis,
    node_size=50,
    alpha=0.8,
    ax=ax
)

# Add labels only for root and level 1 nodes for readability
label_nodes = [node for node in mf_graph.nodes() 
               if distances.get(node, 1000) < 2]
nx.draw_networkx_labels(
    mf_graph,
    pos,
    labels={n: node_labels[n] for n in label_nodes},
    font_size=8,
    ax=ax
)

# Create colorbar
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis,
    norm=plt.Normalize(vmin=min(distances.values()), vmax=max(distances.values())))
sm.set_array([])
#cbar = plt.colorbar(sm, orientation='horizontal', pad=0.05, ax=ax)
#cbar.set_label('Distance from Root', fontsize=12)

# Customize plot
ax.set_title(f"Molecular Function Hierarchy ({len(distances)} terms)", fontsize=16)
ax.axis('off')
fig.tight_layout()
fig.savefig('img/go_molecular_function_hierarchy.png', dpi=300)