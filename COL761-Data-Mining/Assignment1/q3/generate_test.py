import random
import sys

def load_graphs(filename):
    graphs = []
    current_lines = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('t #') or line.startswith('#'):
                if current_lines: graphs.append(current_lines)
                current_lines = []
            current_lines.append(line)
    if current_lines: graphs.append(current_lines)
    return graphs

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_test.py <db_file>")
        sys.exit(1)
        
    db_file = sys.argv[1]
    graphs = load_graphs(db_file)
    
    # Pick 5 random indices
    indices = sorted(random.sample(range(len(graphs)), 5))
    
    print(f"Selected Graph IDs as queries: {indices}")
    
    with open('test_queries.txt', 'w') as f:
        for idx, i in enumerate(indices):
            # Rewrite header to match query format (t # 0, t # 1...)
            # We strip the first line of the original graph (the 't # ID') and replace it
            orig_graph_lines = graphs[i]
            
            # Write new header
            f.write(f"t # {idx}\n")
            
            # Write nodes and edges (skip original header)
            for line in orig_graph_lines[1:]:
                f.write(line)

    # Save the "Correct Answers" for us to check later
    with open('expected_answers.txt', 'w') as f:
        f.write(",".join(map(str, indices)))

if __name__ == "__main__":
    main()