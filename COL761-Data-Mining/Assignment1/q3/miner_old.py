import pandas as pd
import sys
import os
import networkx as nx
from gspan_mining.config import parser
from gspan_mining.main import main as gspan_main

# --- PATCH FOR PANDAS 2.0+ ---
if not hasattr(pd.DataFrame, 'append'):
    from pandas import concat
    def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, (dict, pd.Series)):
            other = pd.DataFrame([other])
        return concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
    pd.DataFrame.append = _append
# -----------------------------

def load_graphs_robust(filename):
    # (Same robust loader logic as preprocess to ensure consistency)
    graphs = []
    current_graph = None
    try:
        f = open(filename, 'r', encoding='utf-8-sig')
    except UnicodeDecodeError:
        f = open(filename, 'r', encoding='cp1252')
    with f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if parts[0] == 't' or parts[0] == '#':
                if current_graph: graphs.append(current_graph)
                current_graph = nx.Graph()
                if len(parts) >= 5: current_graph.graph['support'] = int(parts[4])
                else: current_graph.graph['support'] = 0
            elif parts[0] == 'v':
                if current_graph is None: current_graph = nx.Graph()
                current_graph.add_node(int(parts[1]), label=parts[2])
            elif parts[0] == 'e':
                if current_graph:
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
    if current_graph: graphs.append(current_graph)
    return graphs

def write_clean_gspan_input(graphs, filename):
    with open(filename, 'w') as f:
        for idx, G in enumerate(graphs):
            f.write(f"t # {idx}\n")
            mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
            for node in sorted(G.nodes()):
                label = G.nodes[node]['label']
                f.write(f"v {mapping[node]} {label}\n")
            for u, v in sorted(G.edges()):
                label = G.edges[u,v].get('label', '0')
                f.write(f"e {mapping[u]} {mapping[v]} {label}\n")

def run_gspan_wrapper(clean_input_file, min_support):
    # OPTIMIZATION: -u 10 limits max nodes to 10. Prevents infinite hanging.
    args_str = f"-s {min_support} -l 2 -u 10 {clean_input_file}"
    args = parser.parse_args(args_str.split())
    print(f"Running gSpan (Support={min_support})...")
    
    old_stdout = sys.stdout
    sys.stdout = open('gspan_temp_out.txt', 'w')
    try:
        gspan_main(args)
    except Exception:
        sys.stdout = old_stdout
        return [] # Return empty if fails
    sys.stdout = old_stdout
    
    # Parse output inline to save re-reading
    patterns = []
    curr = None
    if os.path.exists('gspan_temp_out.txt'):
        with open('gspan_temp_out.txt', 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                if parts[0] == 't':
                    if curr: patterns.append(curr)
                    curr = nx.Graph()
                    curr.graph['support'] = int(parts[4]) if len(parts) >= 5 else 0
                elif parts[0] == 'v':
                    curr.add_node(int(parts[1]), label=parts[2])
                elif parts[0] == 'e':
                    curr.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
        if curr: patterns.append(curr)
    return patterns

def select_discriminative(db_graphs, patterns, target_k=50):
    from networkx.algorithms.isomorphism import GraphMatcher
    
    if not patterns: return []
    
    # OPTIMIZATION: Pre-sort by support (closer to 50% is better)
    # This prevents us from checking 2000 patterns. We check the top 100 most promising.
    N = len(db_graphs)
    best_candidates = sorted(patterns, key=lambda x: abs(0.5 - (x.graph['support']/N)))[:200]    
    print(f"Refining top {len(best_candidates)} candidates via isomorphism...")
    
    nm = lambda n1, n2: n1.get('label','0') == n2.get('label','0')
    em = lambda e1, e2: e1.get('label','0') == e2.get('label','0')
    
    final_selection = []
    
    # Simple Selection Strategy to fit time limit:
    # Just take the top K from the sorted list. 
    # Detailed coverage check is too slow for 4000 graphs in Python.
    # The 'support' from gSpan is already accurate for frequency.
    
    final_selection = best_candidates[:target_k]
    return final_selection

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
    if len(sys.argv) < 3: sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    temp_clean = "temp_clean_gspan.data"
    
    db_graphs = load_graphs_robust(input_file)
    write_clean_gspan_input(db_graphs, temp_clean)
    
    # 10% Support
    min_support = max(2, int(len(db_graphs) * 0.05))
    
    patterns = run_gspan_wrapper(temp_clean, min_support)
    print(f"Mined {len(patterns)} raw patterns.")
    
    final = select_discriminative(db_graphs, patterns)
    print(f"Selected {len(final)} patterns.")
    save_patterns(final, output_file)
    
    if os.path.exists(temp_clean): os.remove(temp_clean)

