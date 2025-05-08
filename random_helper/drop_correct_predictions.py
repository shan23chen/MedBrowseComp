import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Drop correct predictions from model output CSV.")
    parser.add_argument("--output_csv", required=True, help="Path to the model output CSV (e.g., nct_876.csv)")
    parser.add_argument("--reference_csv", required=True, help="Path to the reference CSV (e.g., data/Hemonc_dedup_with_all.csv)")
    parser.add_argument("--out", default=None, help="Path for the filtered output CSV. Defaults to input path with '_drop' before .csv")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine output filename
    if args.out is not None:
        out_path = args.out
    else:
        base, ext = os.path.splitext(args.output_csv)
        out_path = f"{base}_drop{ext}"

    # Read model output CSV
    df_out = pd.read_csv(args.output_csv)

    # Find all NCTs where the model got it correct
    correct_ncts = set(
        str(row['extracted_info']).strip()
        for _, row in df_out.iterrows()
        if str(row.get('correct', '')).lower() == 'true' and pd.notna(row.get('extracted_info', ''))
    )

    # Read the reference CSV
    df_ref = pd.read_csv(args.reference_csv)

    # Drop rows from reference where NCT is in correct_ncts
    filtered_df = df_ref[~df_ref['NCT'].astype(str).isin(correct_ncts)]
    filtered_df.to_csv(out_path, index=False)
    print(f"Filtered CSV saved to: {out_path}")


if __name__ == "__main__":
    main()
