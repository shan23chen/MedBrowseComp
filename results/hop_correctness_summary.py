import os
import csv
from collections import defaultdict

RESULTS_DIR = os.path.dirname(__file__)

# Find all CSV files in the directory
csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]

print("\nCorrectness by Hop for Each CSV\n" + "="*50)

for csv_file in csv_files:
    path = os.path.join(RESULTS_DIR, csv_file)
    
    # Initialize all counters
    hop_total = defaultdict(int)          # Total questions per hop
    hop_correct = defaultdict(int)        # Correct answers per hop
    
    hop_na_total = defaultdict(int)       # Questions with NA-like answers per hop
    hop_na_correct = defaultdict(int)     # Correct NA-like answers per hop
    
    hop_real_total = defaultdict(int)     # Non-NA questions per hop
    hop_real_correct = defaultdict(int)   # Correct answers for non-NA questions per hop
    
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        for i, row in enumerate(rows):
            # Assign hop in a repeating pattern: 1,2,3,4,5,1,2,3,4,5,...
            hop = (i % 5) + 1
            
            # Get correct/incorrect status
            val = str(row.get('correct', '')).strip().lower()
            is_correct = val in {'true', '1', 'yes'}
            
            # Get extracted answer and correct answer
            judge_ans = str(row.get('judge_extracted_answer', '')).strip().lower().replace('_', ' ')
            correct_ans = str(row.get('correct_answer', '')).strip().lower().replace('_', ' ')

            # Define what makes an answer NA-like
            na_values = {'na', 'nan', 'not listed'}
            
            # Check for NA-like answers (exact match or contains)
            judge_na = (judge_ans in na_values or 
                       'nan' in judge_ans or 
                       'not listed' in judge_ans or
                       judge_ans.endswith(': na') or
                       judge_ans == '' or       # Empty answer
                       judge_ans == 'none')      # 'None' answer
                       
            correct_na = (correct_ans in na_values or
                         'nan' in correct_ans or
                         'not listed' in correct_ans or
                         correct_ans.endswith(': na') or
                         correct_ans == '' or    # Empty answer
                         correct_ans == 'none')   # 'None' answer
            
            # First, count ALL questions
            hop_total[hop] += 1
            hop_correct[hop] += int(is_correct)
            
            # Count NA questions (where judge's extracted answer is NA-like OR correct answer is NA-like)
            # This is for reporting NA statistics only
            is_na_question = judge_na or correct_na
            if is_na_question:
                hop_na_total[hop] += 1
                if is_correct:
                    hop_na_correct[hop] += 1
                    
            # For REAL accuracy, only exclude questions where the correct answer is NA-like
            # We don't care what the judge extracted for real accuracy calculation
            if not correct_na:
                hop_real_total[hop] += 1
                if is_correct:
                    hop_real_correct[hop] += 1

        # Compute overall real acc
        real_total = sum(hop_real_total.values())
        real_correct = sum(hop_real_correct.values())
    print(f"\nFile: {csv_file}")
    print("Hop\tCorrect/Total\tAccuracy (%)")
    for hop in range(1, 6):  # Show all 5 hops
        correct = hop_correct[hop]
        total = hop_total[hop]
        acc = 100.0 * correct / total if total else 0.0
        print(f"{hop}-hop\t{correct}/{total}\t\t{acc:.1f}")
    print("\nNA/NaN/Not listed correct counts per hop:")
    print("Hop\tNA_Correct/NA_Total\tNA_Accuracy (%)")
    for hop in range(1, 6):  # Show all 5 hops
        na_correct = hop_na_correct[hop]
        na_total = hop_na_total[hop]
        na_acc = 100.0 * na_correct / na_total if na_total else 0.0
        print(f"{hop}-hop\t{na_correct}/{na_total}\t\t{na_acc:.1f}")
    print("\nREAL accuracy (excluding NA-like correct answers):")
    print("Hop\tRealCorrect/RealTotal\tRealAcc (%)")
    for hop in range(1, 6):  # Show all 5 hops
        real_c = hop_real_correct[hop]
        real_t = hop_real_total[hop]
        real_acc = 100.0 * real_c / real_t if real_t else 0.0
        print(f"{hop}-hop\t{real_c}/{real_t}\t\t{real_acc:.1f}")
    print(f"\nOverall REAL accuracy: {real_correct}/{real_total}\t({100.0 * real_correct / real_total if real_total else 0.0:.1f}%)\n")
