import sys

def main():
    # Load Expected IDs (The ones we picked from DB)
    with open('expected_answers.txt', 'r') as f:
        expected_ids = [int(x) for x in f.read().strip().split(',')]

    # Load Candidates
    results = {}
    with open('test_candidates.dat', 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            q_id = int(lines[i].split('#')[1].strip())
            c_line = lines[i+1].split('#')[1].strip()
            if c_line:
                c_ids = set(map(int, c_line.split()))
            else:
                c_ids = set()
            results[q_id] = c_ids

    total_graphs = 40353 # Size of NCI-H23
    
    print(f"{'Query ID':<10} | {'Expected ID':<12} | {'Found?':<8} | {'Candidates':<10} | {'Filtering Ratio':<15}")
    print("-" * 75)

    for i, true_db_id in enumerate(expected_ids):
        candidates = results.get(i, set())
        
        # CHECK 1: RECALL (Did we find the needle in the haystack?)
        found = "YES" if true_db_id in candidates else "NO !!!"
        
        # CHECK 2: FILTERING (Did we remove enough hay?)
        # Filtering Ratio = % of database REMOVED. Higher is better.
        # If candidates = 4000, Ratio = (40000 - 4000) / 40000 = 90%
        ratio = (1 - (len(candidates) / total_graphs)) * 100
        
        print(f"{i:<10} | {true_db_id:<12} | {found:<8} | {len(candidates):<10} | {ratio:.2f}%")

if __name__ == "__main__":
    main()