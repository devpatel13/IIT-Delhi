import sys

# --- YOUR ORIGINAL ROBUST READER ---
def read_auto_decoded(filename):
    """Reads a file with automatic encoding detection."""
    with open(filename, 'rb') as f:
        raw = f.read()
    if raw.startswith(b'\xff\xfe'): return raw.decode('utf-16')
    elif raw.startswith(b'\xef\xbb\xbf'): return raw.decode('utf-8-sig')
    else:
        try: return raw.decode('utf-8')
        except UnicodeDecodeError: return raw.decode('cp1252', errors='ignore')

def load_vectors(filename):
    """Reads space-separated vectors."""
    vectors = []
    content = read_auto_decoded(filename)
    for line in content.splitlines():
        parts = line.strip().split()
        if not parts: continue
        vectors.append(parts)
    return vectors

# --- MAIN UPDATED FOR SUBMISSION FORMAT ---
def main():
    if len(sys.argv) != 4:
        print("Usage: python search.py <db_features> <query_features> <output_file>")
        sys.exit(1)

    db_feat_file = sys.argv[1]
    query_feat_file = sys.argv[2]
    out_file = sys.argv[3]

    db_vecs = load_vectors(db_feat_file)
    query_vecs = load_vectors(query_feat_file)

    with open(out_file, 'w') as f:
        for q_idx, q_vec in enumerate(query_vecs):
            candidates = []
            
            for db_idx, db_vec in enumerate(db_vecs):
                is_match = True
                
                # Logic: Query(1) vs DB(0) is a FAIL
                for q_val, db_val in zip(q_vec, db_vec):
                    if q_val == '1' and db_val == '0':
                        is_match = False
                        break
                
                if is_match:
                    candidates.append(str(db_idx))
            
            # Format: q # ID \n c # ID ID ID
            f.write(f"q # {q_idx}\n")
            f.write(f"c # {' '.join(candidates)}\n")

if __name__ == "__main__":
    main()