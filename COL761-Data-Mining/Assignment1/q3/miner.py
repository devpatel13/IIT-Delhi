import pandas as pd
import sys
import os
import networkx as nx
from gspan_mining.config import parser
from gspan_mining.main import main as gspan_main

# --- PATCH FOR PANDAS 2.0+ SUPPORT ---
if not hasattr(pd.DataFrame, 'append'):
    from pandas import concat
    def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, (dict, pd.Series)):
            other = pd.DataFrame([other])
        return concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
    pd.DataFrame.append = _append
# -------------------------------------

def load_graphs_robust(filename):
    """
    Parses graph files with variable headers (# or t #) and Windows encoding.
    """
    graphs = []
    current_graph = None
    graph_counter = 0

    # Handle Windows BOM (Byte Order Mark) automatically
    try:
        f = open(filename, 'r', encoding='utf-8-sig')
    except UnicodeDecodeError:
        f = open(filename, 'r', encoding='cp1252')

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # Start of Graph (t # ID or just #)
            if parts[0] == 't' or parts[0] == '#':
                if current_graph:
                    graphs.append(current_graph)
                
                # Determine ID
                if len(parts) > 2 and parts[0] == 't':
                     graph_id = parts[2]
                else:
                     graph_id = f"G{graph_counter}"
                
                current_graph = nx.Graph(id=graph_id)
                graph_counter += 1

            # Vertices
            elif parts[0] == 'v':
                if current_graph is None:
                    # Auto-recover from missing header
                    current_graph = nx.Graph(id=f"G{graph_counter}")
                    graph_counter += 1
                current_graph.add_node(int(parts[1]), label=parts[2])
                
            # Edges
            elif parts[0] == 'e':
                if current_graph:
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])

    if current_graph:
        graphs.append(current_graph)
        
    return graphs

def write_clean_gspan_input(graphs, filename):
    """
    Writes graphs to a strictly formatted temporary file that gSpan guarantees to read.
    Format:
    t # N
    v ID LABEL
    e SRC DST LABEL
    """
    with open(filename, 'w') as f:
        for idx, G in enumerate(graphs):
            f.write(f"t # {idx}\n")
            # Sort nodes to ensure valid sequential IDs for gSpan
            # Remap nodes to 0..N-1 to avoid "vertex index out of range" errors
            mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
            
            for node in sorted(G.nodes()):
                label = G.nodes[node]['label']
                f.write(f"v {mapping[node]} {label}\n")
                
            for u, v in sorted(G.edges()):
                label = G.edges[u,v].get('label', '0')
                f.write(f"e {mapping[u]} {mapping[v]} {label}\n")

def run_gspan_wrapper(clean_input_file, min_support):
    """
    Calls the gSpan library on the clean input file.
    """
    # -s: min support, -l: min nodes, -u: max nodes
    args_str = f"-s {min_support} -l 2 -u 10 {clean_input_file}"
    args = parser.parse_args(args_str.split())
    
    print(f"Running gSpan on sanitized input with support={min_support}...")
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = open('gspan_temp_out.txt', 'w')
    
    try:
        gspan_main(args)
    except Exception as e:
        sys.stdout.close()
        sys.stdout = old_stdout
        raise RuntimeError(f"gSpan Internal Error: {e}")
    
    sys.stdout.close()
    sys.stdout = old_stdout
    
    return parse_gspan_output('gspan_temp_out.txt')

def parse_gspan_output(filepath):
    """
    Reads the gSpan output file and reconstructs NetworkX graphs.
    """
    patterns = []
    current_graph = None
    
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('t #'):
                if current_graph: patterns.append(current_graph)
                parts = line.split()
                current_graph = nx.Graph()
                if len(parts) >= 5:
                    current_graph.graph['support_count'] = int(parts[4]) 
                else:
                    current_graph.graph['support_count'] = 0

            elif line.startswith('v'):
                parts = line.split()
                if len(parts) >= 3:
                    current_graph.add_node(int(parts[1]), label=parts[2])
            elif line.startswith('e'):
                parts = line.split()
                if len(parts) >= 4:
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
                
    if current_graph: patterns.append(current_graph)
    return patterns

def check_coverage_and_select(database_graphs, patterns, target_k=50):
    """
    Selects discriminative patterns ensuring coverage.
    """
    from networkx.algorithms.isomorphism import GraphMatcher
    
    if not patterns:
        return []

    nm = lambda n1, n2: n1.get('label', '0') == n2.get('label', '0')
    em = lambda e1, e2: e1.get('label', '0') == e2.get('label', '0')
    
    print("Calculating coverage matrix (this may take a moment)...")
    
    pattern_support_sets = [] 
    
    for i, pat in enumerate(patterns):
        support_set = set()
        for j, G in enumerate(database_graphs):
            # Quick check: node count
            if G.number_of_nodes() < pat.number_of_nodes():
                continue
                
            GM = GraphMatcher(G, pat, node_match=nm, edge_match=em)
            if GM.subgraph_is_isomorphic():
                support_set.add(j)
        pattern_support_sets.append(support_set)
        
        if i % 10 == 0:
            print(f"Checked pattern {i}/{len(patterns)}", end='\r')
            
    print("\n")

    N = len(database_graphs)
    scores = []
    for i, support_set in enumerate(pattern_support_sets):
        support_fraction = len(support_set) / N
        score = support_fraction * (1 - support_fraction) 
        scores.append((score, i))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    selected_indices = [idx for score, idx in scores[:target_k]]
    
    return [patterns[i] for i in selected_indices]

def save_patterns(patterns, output_path):
    with open(output_path, 'w') as f:
        for i, G in enumerate(patterns):
            f.write(f"t # {i}\n")
            for node in sorted(G.nodes()):
                f.write(f"v {node} {G.nodes[node]['label']}\n")
            for u, v in sorted(G.edges()):
                label = G.edges[u,v].get('label', '0')
                f.write(f"e {u} {v} {label}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python miner.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    temp_clean_file = "temp_clean_input.data"
    
    # 1. Load Data Robustly
    print("Loading and sanitizing input database...")
    try:
        db_graphs = load_graphs_robust(input_file)
    except Exception as e:
        print(f"Error loading graphs: {e}")
        sys.exit(1)

    if not db_graphs:
        print("Error: No graphs found.")
        sys.exit(1)
        
    # 2. Write to Clean Format for gSpan
    write_clean_gspan_input(db_graphs, temp_clean_file)
    
    # 3. Run gSpan on Clean Data
    # Support: Ensure at least 2 graphs, or 10%
    min_support = max(2, int(len(db_graphs) * 0.1))
    
    try:
        all_patterns = run_gspan_wrapper(temp_clean_file, min_support)
        print(f"Mined {len(all_patterns)} frequent patterns.")
        
        # 4. Select Best Patterns
        final_patterns = check_coverage_and_select(db_graphs, all_patterns, target_k=50)
        
        print(f"Selected {len(final_patterns)} discriminative subgraphs.")
        save_patterns(final_patterns, output_file)
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_clean_file):
            os.remove(temp_clean_file)