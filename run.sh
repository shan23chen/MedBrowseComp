#!/bin/bash

python process_NCT_predictions.py data/final50.csv --task filled50 --use_judge --use_tools --output results_with_judge.csv --model gemini-2.5-pro-exp-03-25

python process_NCT_predictions.py data/final50.csv --task filled50 --use_judge --output results_with_judge.csv --model sonar-deep-research

# INPUT="data/nct_876_drop.csv"
# MODEL="gemini-2.5-pro-preview-03-25"

# DIR="results/vibe/${MODEL}"

# mkdir -p "$DIR"

# for TASK in track_second_authors track_pmids track_start_date
# do
#     python process_NCT_predictions.py "$INPUT" --task $TASK --model $MODEL --output "$DIR/${TASK}_${MODEL}_no_tools.csv"
#     python process_NCT_predictions.py "$INPUT" --task $TASK --model $MODEL --use_tools --output "$DIR/${TASK}_${MODEL}_with_tools.csv"
# done