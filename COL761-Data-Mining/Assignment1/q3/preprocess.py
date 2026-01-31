import sys
import os
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

def parse_dataset(file_path):
    """
    Parses the specific graph format provided in the assignment.
    Format:
    # (Graph separator, implicitly starts a new graph)
    v <node_id> <label>
    e <src> <dst> <label>
    """
    graphs = []
    current_graph = None
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line == "#":
            if current_graph is not None:
                graphs.append(current_graph)
            current_graph = nx.Graph() # Undirected graph
            
        elif line.startswith("v"):
            parts = line.split()
            node_id = int(parts[1])
            node_label = parts[2]
            current_graph.add_node(node_id, label=node_label)
            
        elif line.startswith("e"):
            parts = line.split()
            u = int(parts[1])
            v = int(parts[2])
            edge_label = parts[3]
            # NetworkX Graph is undirected, so adding (u, v) and (v, u) is fine, 
            # it effectively updates the same edge.
            current_graph.add_edge(u, v, label=edge_label)
            
    # Append the last graph if file doesn't end with newline/hash
    if current_graph is not None and len(current_graph.nodes) > 0:
        graphs.append(current_graph)
        
    return graphs

def save_for_gspan(graphs, output_path):
    """
    Saves the list of NetworkX graphs in the standard gSpan input format.
    t # <graph_id>
    v <node_id> <label>
    e <u_id> <v_id> <label>
    """
    with open(output_path, 'w') as f:
        for gid, G in enumerate(graphs):
            f.write(f"t # {gid}\n")
            
            # Nodes need to be re-indexed to ensure they are continuous 0..N-1
            # (Though in your data they seem to be already, it's safer to remap)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            
            for node in G.nodes():
                label = G.nodes[node]['label']
                f.write(f"v {mapping[node]} {label}\n")
                
            for u, v in G.edges():
                label = G.edges[u, v]['label']
                f.write(f"e {mapping[u]} {mapping[v]} {label}\n")
                
    print(f"Saved {len(graphs)} unique graphs to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_file_path> <output_file_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Processing {input_path}...")
    
    # 1. Parse
    all_graphs = parse_dataset(input_path)
    print(f"Total graphs loaded: {len(all_graphs)}")
    
    # 2. Deduplicate using Weisfeiler-Lehman Hash
    unique_graphs = []
    seen_hashes = set()
    
    for G in all_graphs:
        # 'node_attr' must match the attribute name used in parse_dataset ('label')
        # WL hash is robust for isomorphism checks on labeled graphs
        ghash = weisfeiler_lehman_graph_hash(G, node_attr='label')
        
        if ghash not in seen_hashes:
            seen_hashes.add(ghash)
            unique_graphs.append(G)
            
    print(f"Unique graphs found: {len(unique_graphs)}")
    
    # 3. Save for gSpan
    save_for_gspan(unique_graphs, output_path)

if __name__ == "__main__":
    main()