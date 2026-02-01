import pandas as pd
import sys
import os
import networkx as nx
from collections import Counter
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

# --- ROBUST LOADER ---
def load_graphs_robust(filename):
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
            if parts[0] == '#' or parts[0] == 't':
                if current_graph: graphs.append(current_graph)
                current_graph = nx.Graph()
                current_graph.graph['support'] = int(parts[4]) if len(parts) >= 5 else 0
            elif parts[0] == 'v':
                if current_graph is None: current_graph = nx.Graph()
                current_graph.add_node(int(parts[1]), label=parts[2])
            elif parts[0] == 'e':
                if current_graph:
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
    if current_graph: graphs.append(current_graph)
    return graphs

# --- GSPAN WRAPPER ---
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
    # Mine patterns with at least 2 nodes (standard)
    args_str = f"-s {min_support} -l 2 -u 10 {clean_input_file}"
    args = parser.parse_args(args_str.split())
    
    print(f"Running gSpan (Support={min_support})...")
    old_stdout = sys.stdout
    sys.stdout = open('gspan_temp_out.txt', 'w')
    try:
        gspan_main(args)
    except Exception:
        sys.stdout = old_stdout
        return [] 
    sys.stdout = old_stdout
    
    # Parse output
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

# --- NEW: ATOMIC INJECTION (The Fix from the Paper) ---
def get_atomic_patterns(db_graphs, top_k=20):
    """
    Creates single-node graphs for the most frequent atoms (C, O, N, etc.).
    This ensures that even small queries match *something*.
    """
    print("Generating Atomic Patterns (Base Coverage)...")
    all_labels = []
    for G in db_graphs:
        for _, data in G.nodes(data=True):
            all_labels.append(data.get('label', 'C'))
    
    # Get top K atoms
    common_labels = [label for label, count in Counter(all_labels).most_common(top_k)]
    
    atomic_patterns = []
    for label in common_labels:
        G = nx.Graph()
        G.add_node(0, label=label)
        # Give fake high support so they are prioritized
        G.graph['support'] = 99999 
        atomic_patterns.append(G)
        
    return atomic_patterns

def select_discriminative(db_graphs, gspan_patterns, target_k=150):
    N = len(db_graphs)
    
    # 1. Atomic Patterns (The Safety Net)
    atomic = get_atomic_patterns(db_graphs, top_k=20)
    
    # 2. Discriminative Patterns (The Filter)
    # Filter out very rare patterns (< 5%)
    valid_gspan = [p for p in gspan_patterns if p.graph['support'] > (N * 0.05)]
    # Sort by split ratio (closer to 0.5 is better)
    discriminative = sorted(valid_gspan, key=lambda x: abs(0.5 - (x.graph['support']/N)))
    
    selection = []
    # Add atoms first
    selection.extend(atomic)
    
    # Add complex patterns
    for p in discriminative:
        if len(selection) >= target_k:
            break
        selection.append(p)
            
    return selection

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
    
    # Mine (5% support)
    min_support = max(2, int(len(db_graphs) * 0.05))
    gspan_patterns = run_gspan_wrapper(temp_clean, min_support)
    
    # Select best features (Atoms + Discriminative)
    final = select_discriminative(db_graphs, gspan_patterns, target_k=150)
    
    print(f"Selected {len(final)} features (Atomic + Discriminative).")
    save_patterns(final, output_file)
    
    if os.path.exists(temp_clean): os.remove(temp_clean)