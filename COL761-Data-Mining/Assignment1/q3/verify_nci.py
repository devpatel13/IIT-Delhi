import sys

# Reference values from your friend's image
FRIEND_RESULTS = {
    0: 1745,
    1: 3592,
    2: 461,
    5: 4307,
    10: 728,
    15: 23,
    20: 2736,
    25: 1633,
    30: 3167,
    35: 4998,
    40: 1556,
    45: 1859,
    49: 3395,
    27: 2499,
    32: 809,
    17: 675,
    41: 3134,
    33: 483
}

def main():
    my_results = {}
    
    try:
        with open('candidates_nci_visible.dat', 'r') as f:
            lines = f.readlines()
            # Process in pairs: (Query Header, Candidate List)
            for i in range(0, len(lines), 2):
                if i+1 >= len(lines): break
                
                header = lines[i].strip() # e.g. "t # 0" or "q # 0"
                body = lines[i+1].strip() # e.g. "c # 12 45..."
                
                parts = header.split('#')
                if len(parts) < 2: continue
                q_id = int(parts[1].strip())
                
                # Check for "c #" prefix in body
                if '#' in body:
                    c_list = body.split('#')[1].strip()
                    count = len(c_list.split()) if c_list else 0
                else:
                    count = 0
                    
                my_results[q_id] = count
                
    except FileNotFoundError:
        print("Error: candidates_nci_visible.dat not found.")
        sys.exit(1)

    print(f"{'Q_ID':<5} | {'Friend':<8} | {'You':<8} | {'Status'}")
    print("-" * 45)
    
    sorted_ids = sorted(FRIEND_RESULTS.keys())
    
    better = 0
    total = 0
    
    for qid in sorted_ids:
        if qid not in my_results:
            print(f"{qid:<5} | {FRIEND_RESULTS[qid]:<8} | {'MISSING':<8} | ?")
            continue
            
        friend_n = FRIEND_RESULTS[qid]
        my_n = my_results[qid]
        total += 1
        
        diff = my_n - friend_n
        
        if my_n <= friend_n:
            # If your set is smaller or equal, that's GREAT (Better Filtering)
            # ...assuming Recall is 100% (which we verified earlier)
            status = "✅ MATCH/BETTER"
            better += 1
        else:
            status = f"❌ WORSE (+{diff})"
            
        print(f"{qid:<5} | {friend_n:<8} | {my_n:<8} | {status}")

    print("-" * 45)
    print(f"Summary: You beat or matched your friend on {better}/{total} queries.")

if __name__ == "__main__":
    main()