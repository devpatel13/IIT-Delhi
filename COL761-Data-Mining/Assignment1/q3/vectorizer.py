import sys
import networkx as nx
from networkx.algorithms import isomorphism
from collections import Counter

# --- YOUR ORIGINAL ROBUST PARSING LOGIC ---
def parse_graph_file(filename):
    """Parses graph files with variable headers and encoding fixes."""
    graphs = []
    current_graph = None
    graph_counter = 0

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
                graph_id = parts[2] if len(parts) > 2 and parts[0] == 't' else f"G{graph_counter}"
                current_graph = nx.Graph(id=graph_id)
                graph_counter += 1
            elif parts[0] == 'v':
                if current_graph is None:
                    current_graph = nx.Graph(id=f"G{graph_counter}")
                    graph_counter += 1
                current_graph.add_node(int(parts[1]), label=parts[2])
            elif parts[0] == 'e':
                if current_graph:
                    current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])

    if current_graph: graphs.append(current_graph)
    return graphs

def precompute_metadata(graph):
    labels = [data.get('label', '0') for _, data in graph.nodes(data=True)]
    return {
        'count': graph.number_of_nodes(),
        'label_counts': Counter(labels),
        'graph': graph
    }

def get_feature_vector(g_meta, patterns_meta, node_matcher):
    vector = []
    for p_meta in patterns_meta:
        # OPTIMIZATION 1: Node Count
        if p_meta['count'] > g_meta['count']:
            vector.append('0')
            continue

        # OPTIMIZATION 2: Label Histogram
        possible = True
        for label, count in p_meta['label_counts'].items():
            if g_meta['label_counts'].get(label, 0) < count:
                possible = False
                break
        if not possible:
            vector.append('0')
            continue
            
        # OPTIMIZATION 3: Isomorphism
        GM = isomorphism.GraphMatcher(g_meta['graph'], p_meta['graph'], node_match=node_matcher)
        if GM.subgraph_is_isomorphic():
            vector.append('1')
        else:
            vector.append('0')
    
    # CHANGE: Return space-separated string "0 1 0 1" (Numpy style)
    return " ".join(vector)

# --- MAIN UPDATED FOR SUBMISSION FORMAT ---
def main():
    if len(sys.argv) != 4:
        print("Usage: python vectorizer.py <input_graphs> <patterns_file> <output_features_file>")
        sys.exit(1)

    db_file = sys.argv[1]
    patterns_file = sys.argv[2]
    out_file = sys.argv[3]

    try:
        raw_db_graphs = parse_graph_file(db_file)
        raw_patterns = parse_graph_file(patterns_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    db_metas = [precompute_metadata(g) for g in raw_db_graphs]
    patterns_metas = [precompute_metadata(p) for p in raw_patterns]
    node_matcher = isomorphism.categorical_node_match('label', 'X')

    # Write to output file directly
    with open(out_file, 'w') as f:
        for g_meta in db_metas:
            vec = get_feature_vector(g_meta, patterns_metas, node_matcher)
            f.write(f"{vec}\n")

if __name__ == "__main__":
    main()