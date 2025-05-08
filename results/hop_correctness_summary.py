import os
import csv
from collections import defaultdict

RESULTS_DIR = os.path.dirname(__file__)

# Find all CSV files in the directory
csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]

print("\nCorrectness by Hop for Each CSV in results121\n" + "="*50)

for csv_file in csv_files:
    path = os.path.join(RESULTS_DIR, csv_file)
    hop_correct = defaultdict(int)
    hop_total = defaultdict(int)
    
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        n = len(rows)
        # Assume 5 hops, each hop has n//5 questions
        hop_size = n // 5
        for i, row in enumerate(rows):
            hop = (i // hop_size) + 1
            # Clamp to 5
            hop = min(hop, 5)
            # Accept True/true/1 as correct
            val = str(row.get('correct', '')).strip().lower()
            is_correct = val in {'true', '1', 'yes'}
            hop_correct[hop] += int(is_correct)
            hop_total[hop] += 1
    print(f"\nFile: {csv_file}")
    print("Hop\tCorrect/Total\tAccuracy (%)")
    for hop in range(1, 6):
        correct = hop_correct[hop]
        total = hop_total[hop]
        acc = 100.0 * correct / total if total else 0.0
        print(f"{hop}-hop\t{correct}/{total}\t\t{acc:.1f}")
