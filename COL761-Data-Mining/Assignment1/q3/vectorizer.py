import sys
import networkx as nx
from networkx.algorithms import isomorphism
from collections import Counter
from multiprocessing import Pool, cpu_count
import time

# --- 1. PARSING LOGIC (Robust & Unchanged) ---
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

# --- 2. WORKER FUNCTION (Unchanged Logic) ---
def process_single_graph(args):
    g_meta, patterns_meta = args
    node_matcher = isomorphism.categorical_node_match('label', 'X')
    
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
            
    return " ".join(vector)

# --- 3. MAIN (Updated with Progress Bar) ---
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
    
    tasks = [(g, patterns_metas) for g in db_metas]
    total_graphs = len(tasks)
    
    n_cores = cpu_count()
    print(f"Vectorizing {total_graphs} graphs using {n_cores} parallel cores...")
    
    start_time = time.time()
    processed_count = 0
    
    # Use Pool with imap to track progress
    with Pool(n_cores) as p:
        # imap returns an iterator that yields results in order as they complete
        result_iterator = p.imap(process_single_graph, tasks, chunksize=100)
        
        with open(out_file, 'w') as f:
            for vec in result_iterator:
                f.write(f"{vec}\n")
                
                processed_count += 1
                
                # Update progress every 500 graphs (avoids printing too fast)
                if processed_count % 500 == 0 or processed_count == total_graphs:
                    elapsed = time.time() - start_time
                    percent = (processed_count / total_graphs) * 100
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    
                    # \r overwrites the current line
                    sys.stderr.write(f"\rProgress: {processed_count}/{total_graphs} ({percent:.1f}%) - {speed:.1f} graphs/sec")
                    sys.stderr.flush()

    # Final newline
    sys.stderr.write("\n")
    print(f"Done in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()