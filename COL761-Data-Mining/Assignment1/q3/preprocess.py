import sys
import os
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

def parse_dataset_robust(file_path):
    """
    Robustly parses graph files (handles #, t #, and Windows encoding).
    """
    graphs = []
    current_graph = None
    graph_counter = 0

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    # Handle Windows/Linux encoding differences
    try:
        f = open(file_path, 'r', encoding='utf-8-sig')
    except UnicodeDecodeError:
        f = open(file_path, 'r', encoding='cp1252')

    with f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            
            # Start of Graph (t # ID or just #)
            if parts[0] == '#' or parts[0] == 't':
                if current_graph: graphs.append(current_graph)
                current_graph = nx.Graph() 
                
            elif parts[0] == 'v':
                if current_graph is None: current_graph = nx.Graph()
                current_graph.add_node(int(parts[1]), label=parts[2])
                
            elif parts[0] == 'e':
                if current_graph:
                    # Undirected: NetworkX handles (u,v) == (v,u) automatically
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
    
    if current_graph and len(current_graph.nodes) > 0:
        graphs.append(current_graph)
    return graphs

def save_for_gspan(graphs, output_path):
    with open(output_path, 'w') as f:
        for gid, G in enumerate(graphs):
            f.write(f"t # {gid}\n")
            mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
            for node in sorted(G.nodes()):
                label = G.nodes[node]['label']
                f.write(f"v {mapping[node]} {label}\n")
            for u, v in sorted(G.edges()):
                label = G.edges[u, v].get('label', '0')
                f.write(f"e {mapping[u]} {mapping[v]} {label}\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Processing {input_path}...")
    all_graphs = parse_dataset_robust(input_path)
    print(f"Total graphs loaded: {len(all_graphs)}")
    
    # Deduplicate using WL Hash (Your logic: Correct & Accurate)
    unique_graphs = []
    seen_hashes = set()
    
    print("Deduplicating...")
    for G in all_graphs:
        # WL hash is robust for isomorphism checks on labeled graphs
        ghash = weisfeiler_lehman_graph_hash(G, node_attr='label')
        if ghash not in seen_hashes:
            seen_hashes.add(ghash)
            unique_graphs.append(G)
            
    print(f"Unique graphs found: {len(unique_graphs)}")
    save_for_gspan(unique_graphs, output_path)

if __name__ == "__main__":
    main()


