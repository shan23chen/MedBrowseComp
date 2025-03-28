import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
import json
from datetime import datetime
from gemini_inference import GEMINI_MODELS
from process_NCT_predictions import process_nct_csv  

def run_model_comparisons(
    csv_path: str,
    output_dir: str = "results",
    max_workers: int = 4,
    test_mode: bool = False,
    n: int = None,
    models_to_test: List[str] = None
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
    
    # Dictionary to store results
    results = {}
    run_dir = "./results"
    # Run all configurations
    for model_name in models_to_test:
        for use_tools in [False, True]:
            config_name = f"{model_name}_{'with' if use_tools else 'without'}_tools"
            print(f"\n{'='*80}\nRunning configuration: {config_name}\n{'='*80}")
            
            # Create output path for this configuration with clearer naming
            search_suffix = "_with_search" if use_tools else "_no_search"
            output_csv = os.path.join(run_dir, f"{model_name}{search_suffix}.csv")
            
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
            
            # Store results
            results[config_name] = {
                'model': model_name,
                'use_tools': use_tools,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': total_count,
                'output_csv': output_csv
            }
            
            print(f"Configuration {config_name}: Accuracy {correct_count}/{total_count} ({accuracy:.2%})")
    
    # Save overall results to JSON
    results_json_path = os.path.join(run_dir, f"comparison_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Create a summary CSV with all results
    summary_df = pd.DataFrame([
        {
            'Model': info['model'],
            'Search': 'Yes' if info['use_tools'] else 'No',
            'Accuracy': info['accuracy'],
            'Correct': info['correct_count'],
            'Total': info['total_count'],
            'CSV_Path': info['output_csv']
        }
        for config, info in results.items()
    ])
    
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
    
    # Create a DataFrame from results
    data = []
    for config_name, config_results in results.items():
        data.append({
            'Configuration': config_name,
            'Model': config_results['model'],
            'Tools': 'With Tools' if config_results['use_tools'] else 'Without Tools',
            'Accuracy': config_results['accuracy'],
        })
    
    df = pd.DataFrame(data)
    
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create accuracy comparison by model and tools
    bar_plot = sns.barplot(x='Model', y='Accuracy', hue='Tools', data=df, palette="viridis")
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add accuracy values on top of bars
    for i, p in enumerate(bar_plot.patches):
        height = p.get_height()
        bar_plot.text(p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.2%}',
                ha="center", fontsize=10)
    
    plt.tight_layout()
    # Save to the run directory
    plt.savefig(os.path.join(output_dir, f"run_{timestamp}/model_comparison.png"), dpi=300)
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
        bar_plot.text(max(width - 0.1, 0.05),
                p.get_y() + p.get_height()/2.,
                f'{width:.2%}',
                ha="right", va="center", fontsize=10, color="white", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"run_{timestamp}/model_ranking.png"), dpi=300)
    plt.close()
    
    # Create a tools impact analysis
    plt.figure(figsize=(10, 6))
    
    # Create a DataFrame showing the impact of tools for each model
    tool_impact = []
    for model in df['Model'].unique():
        no_tools = df[(df['Model'] == model) & (df['Tools'] == 'Without Tools')]['Accuracy'].values[0]
        with_tools = df[(df['Model'] == model) & (df['Tools'] == 'With Tools')]['Accuracy'].values[0]
        impact = with_tools - no_tools
        tool_impact.append({
            'Model': model,
            'Impact': impact,
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
        if width >= 0:
            x_pos = max(width + 0.01, 0.02)
            h_align = "left"
        else:
            x_pos = min(width - 0.01, -0.02)
            h_align = "right"
        
        bar_plot.text(x_pos,
                p.get_y() + p.get_height()/2.,
                f'{width:+.2%}',
                ha=h_align, va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"run_{timestamp}/tools_impact.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/run_{timestamp}")

def main():
    parser = argparse.ArgumentParser(description="Compare model performance across configurations")
    parser.add_argument("csv_path", help="Path to CSV file with evidence and NCT columns")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 8 examples")
    parser.add_argument("-n", type=int, help="Process only the first n rows (for testing)")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--visualize_only", help="Path to existing JSON results file to visualize")
    
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
        
        visualize_results(results, vis_dir, "")
    else:
        # Run comparisons
        results = run_model_comparisons(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            max_workers=args.threads,
            test_mode=args.test,
            n=args.n,
            models_to_test=args.models
        )
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualize_results(results, args.output_dir, timestamp)

if __name__ == "__main__":
    main()