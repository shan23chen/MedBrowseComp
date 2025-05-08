import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
import json
from datetime import datetime
import statistics
from gemini_inference import GEMINI_MODELS
from process_NCT_predictions import process_nct_csv  

def run_model_comparisons(
    csv_path: str,
    output_dir: str = "results",
    max_workers: int = 4,
    test_mode: bool = False,
    n: int = None,
    models_to_test: List[str] = None,
    runs: int = 1
) -> Dict:
    """
    Run comparisons across all model configurations
    
    Args:
        csv_path: Path to CSV file with evidence and NCT columns
        output_dir: Directory to save results
        max_workers: Number of parallel threads
        test_mode: Whether to run in test mode (only 8 examples)
        n: Number of rows to process
        models_to_test: List of specific models to test (if None, test all)
        runs: Number of times to repeat each configuration (default: 1)
        
    Returns:
        Dictionary with results for each configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If not specified, use all available models
    if not models_to_test:
        models_to_test = list(GEMINI_MODELS.keys())
    
    # Validate models
    for model in models_to_test:
        if model not in GEMINI_MODELS:
            print(f"Invalid model name: {model}. Skipping.")
            models_to_test.remove(model)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Run all configurations
    for model_name in models_to_test:
        for use_tools in [False, True]:
            config_name = f"{model_name}_{'with' if use_tools else 'without'}_tools"
            print(f"\n{'='*80}\nRunning configuration: {config_name}\n{'='*80}")
            
            # Lists to store multiple run results
            all_run_accuracies = []
            all_run_results = []
            
            # Run multiple times if requested
            for run_idx in range(runs):
                print(f"\nRun {run_idx + 1}/{runs}")
                
                # Create output path for this configuration with clearer naming
                search_suffix = "_with_search" if use_tools else "_no_search"
                run_suffix = f"_run{run_idx + 1}" if runs > 1 else ""
                output_csv = os.path.join(run_dir, f"{model_name}{search_suffix}{run_suffix}.csv")
                
                # Run the process_nct_csv function
                processed_results = process_nct_csv(
                    csv_path=csv_path,
                    model_name=model_name,
                    use_tools=use_tools,
                    max_workers=max_workers,
                    output_path=output_csv,
                    test_mode=test_mode,
                    n=n
                )
                
                # Calculate accuracy
                correct_count = sum(1 for result in processed_results if result['correct'])
                total_count = len(processed_results)
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                # Store this run's results
                run_result = {
                    'run': run_idx + 1,
                    'model': model_name,
                    'use_tools': use_tools,
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_count': total_count,
                    'output_csv': output_csv
                }
                
                all_run_accuracies.append(accuracy)
                all_run_results.append(run_result)
                
                print(f"  Run {run_idx + 1}: Accuracy {correct_count}/{total_count} ({accuracy:.2%})")
            
            # Calculate average metrics across runs
            avg_accuracy = statistics.mean(all_run_accuracies)
            std_dev = statistics.stdev(all_run_accuracies) if runs > 1 else 0
            
            # Store aggregated results
            results[config_name] = {
                'model': model_name,
                'use_tools': use_tools,
                'accuracy': avg_accuracy,
                'accuracy_std_dev': std_dev,
                'runs': runs,
                'individual_runs': all_run_results,
                'all_accuracies': all_run_accuracies
            }
            
            if runs > 1:
                print(f"\nConfiguration {config_name}: Average Accuracy {avg_accuracy:.2%} (±{std_dev:.2%})")
            else:
                print(f"\nConfiguration {config_name}: Accuracy {avg_accuracy:.2%}")
    
    # Save overall results to JSON
    results_json_path = os.path.join(run_dir, f"comparison_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Create a summary CSV with all results
    summary_rows = []
    for config, info in results.items():
        summary_row = {
            'Model': info['model'],
            'Search': 'Yes' if info['use_tools'] else 'No',
            'Avg_Accuracy': info['accuracy'],
            'Std_Dev': info['accuracy_std_dev'],
            'Runs': info['runs'],
        }
        
        # Add individual run accuracies if there are multiple runs
        if runs > 1:
            for i, acc in enumerate(info['all_accuracies']):
                summary_row[f'Run_{i+1}_Accuracy'] = acc
        
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(run_dir, f"summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\nAll results saved to {results_json_path}")
    return results

def visualize_results(results: Dict, output_dir: str = "results", timestamp: str = None):
    """
    Create visualizations for comparison results
    
    Args:
        results: Dictionary with results for each configuration
        output_dir: Directory to save visualizations
        timestamp: Optional timestamp for file naming
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    vis_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a DataFrame from results
    data = []
    for config_name, config_results in results.items():
        data.append({
            'Configuration': config_name,
            'Model': config_results['model'],
            'Tools': 'With Tools' if config_results['use_tools'] else 'Without Tools',
            'Accuracy': config_results['accuracy'],
            'Std_Dev': config_results.get('accuracy_std_dev', 0),
            'Runs': config_results.get('runs', 1)
        })
    
    df = pd.DataFrame(data)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create accuracy comparison by model and tools
    plt.figure(figsize=(12, 8))
    
    # Add error bars if multiple runs were performed
    if any(x > 1 for x in df['Runs']):
        bar_plot = sns.barplot(x='Model', y='Accuracy', hue='Tools', data=df, palette="viridis", 
                               errorbar=('ci', 95), errwidth=1.5, capsize=0.1)
        plt.title('Model Accuracy Comparison (with 95% CI)', fontsize=16)
    else:
        bar_plot = sns.barplot(x='Model', y='Accuracy', hue='Tools', data=df, palette="viridis")
        plt.title('Model Accuracy Comparison', fontsize=16)
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add accuracy values on top of bars
    for i, p in enumerate(bar_plot.patches):
        height = p.get_height()
        row = df.iloc[i // 2] if len(df) * 2 == len(bar_plot.patches) else df.iloc[i]
        
        if row['Runs'] > 1:
            label = f'{height:.2%}\n±{row["Std_Dev"]:.2%}'
        else:
            label = f'{height:.2%}'
            
        bar_plot.text(p.get_x() + p.get_width()/2.,
                height + 0.01,
                label,
                ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Create a horizontal bar chart sorted by accuracy
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('Accuracy', ascending=True)
    
    # Combine model and tools for the y-axis
    df_sorted['Model_Tools'] = df_sorted['Model'] + ' (' + df_sorted['Tools'] + ')'
    
    # Create the horizontal bar chart
    bar_plot = sns.barplot(x='Accuracy', y='Model_Tools', data=df_sorted, palette="viridis")
    plt.title('Models Ranked by Accuracy', fontsize=16)
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Model Configuration', fontsize=14)
    plt.xlim(0, 1)
    
    # Add accuracy values inside bars
    for i, p in enumerate(bar_plot.patches):
        width = p.get_width()
        row = df_sorted.iloc[i]
        
        if row['Runs'] > 1:
            label = f'{width:.2%} ±{row["Std_Dev"]:.2%}'
        else:
            label = f'{width:.2%}'
            
        bar_plot.text(max(width - 0.1, 0.05),
                p.get_y() + p.get_height()/2.,
                label,
                ha="right", va="center", fontsize=10, color="white", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "model_ranking.png"), dpi=300)
    plt.close()
    
    # Create a tools impact analysis
    plt.figure(figsize=(10, 6))
    
    # Create a DataFrame showing the impact of tools for each model
    tool_impact = []
    for model in df['Model'].unique():
        no_tools_row = df[(df['Model'] == model) & (df['Tools'] == 'Without Tools')]
        with_tools_row = df[(df['Model'] == model) & (df['Tools'] == 'With Tools')]
        
        if len(no_tools_row) > 0 and len(with_tools_row) > 0:
            no_tools = no_tools_row['Accuracy'].values[0]
            with_tools = with_tools_row['Accuracy'].values[0]
            impact = with_tools - no_tools
            
            # Calculate combined standard deviation if multiple runs
            if any(x > 1 for x in df['Runs']):
                no_tools_std = no_tools_row['Std_Dev'].values[0]
                with_tools_std = with_tools_row['Std_Dev'].values[0]
                # Combined standard deviation for difference
                combined_std = (no_tools_std**2 + with_tools_std**2)**0.5
            else:
                combined_std = 0
                
            tool_impact.append({
                'Model': model,
                'Impact': impact,
                'Std_Dev': combined_std,
                'Positive': impact >= 0
            })
    
    impact_df = pd.DataFrame(tool_impact)
    impact_df = impact_df.sort_values('Impact', ascending=True)
    
    # Create the horizontal bar chart for tool impact
    colors = ['green' if x else 'red' for x in impact_df['Positive']]
    bar_plot = sns.barplot(x='Impact', y='Model', data=impact_df, palette=colors)
    
    plt.title('Impact of Tools on Accuracy by Model', fontsize=16)
    plt.xlabel('Accuracy Difference (With Tools - Without Tools)', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add impact values inside bars
    for i, p in enumerate(bar_plot.patches):
        width = p.get_width()
        row = impact_df.iloc[i]
        
        if row['Std_Dev'] > 0:
            label = f'{width:+.2%} ±{row["Std_Dev"]:.2%}'
        else:
            label = f'{width:+.2%}'
            
        if width >= 0:
            x_pos = max(width + 0.01, 0.02)
            h_align = "left"
        else:
            x_pos = min(width - 0.01, -0.02)
            h_align = "right"
        
        bar_plot.text(x_pos,
                p.get_y() + p.get_height()/2.,
                label,
                ha=h_align, va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "tools_impact.png"), dpi=300)
    plt.close()
    
    # If multiple runs, create a box plot to show distribution
    if any(x > 1 for x in df['Runs']):
        # Create a DataFrame for box plot
        box_data = []
        for config, info in results.items():
            for acc in info.get('all_accuracies', []):
                box_data.append({
                    'Model': info['model'],
                    'Tools': 'With Tools' if info['use_tools'] else 'Without Tools',
                    'Accuracy': acc
                })
        
        if box_data:
            box_df = pd.DataFrame(box_data)
            
            plt.figure(figsize=(14, 8))
            box_plot = sns.boxplot(x='Model', y='Accuracy', hue='Tools', data=box_df, palette="viridis")
            
            plt.title('Accuracy Distribution Across Runs', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            
            plt.savefig(os.path.join(vis_dir, "accuracy_distribution.png"), dpi=300)
            plt.close()
    
    print(f"Visualizations saved to {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare model performance across configurations")
    parser.add_argument("csv_path", help="Path to CSV file with evidence and NCT columns")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 8 examples")
    parser.add_argument("-n", type=int, help="Process only the first n rows (for testing)")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--visualize_only", help="Path to existing JSON results file to visualize")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to repeat each configuration")
    
    args = parser.parse_args()
    
    if args.visualize_only:
        # Load existing results and visualize
        with open(args.visualize_only, 'r') as f:
            results = json.load(f)
        
        # Create timestamp for this visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualization directory
        vis_dir = os.path.join(args.output_dir, f"visualization_{timestamp}")
        os.makedirs(vis_dir, exist_ok=True)
        
        visualize_results(results, vis_dir, timestamp)
    else:
        # Run comparisons
        results = run_model_comparisons(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            max_workers=args.threads,
            test_mode=args.test,
            n=args.n,
            models_to_test=args.models,
            runs=args.runs
        )
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualize_results(results, args.output_dir, timestamp)

if __name__ == "__main__":
    main()