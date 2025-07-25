#!/usr/bin/env python3
"""
Script to evaluate MemGPT hypothesis files across different beta values.
This script runs the evaluation on all beta values from 0.0 to 1.0 and collects results.
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
import numpy as np

def run_evaluation(metric_model, hyp_file, ref_file):
    """Run the evaluation script and return the results."""
    script_path = "/home/samer/Documents/LAU/Research/focus_memgpt/Focused-MemGPT/paper_experiments/longmemeval/evaluation_scripts/evaluate_qa.py"#"evaluation_scripts/evaluate_qa.py"
    
    cmd = [sys.executable, script_path, metric_model, hyp_file, ref_file]
    
    print(f"Running evaluation for {hyp_file}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to extract accuracy results
        lines = result.stdout.strip().split('\n')
        results = {}
        
        for line in lines:
            if line.startswith('Accuracy:'):
                overall_acc = float(line.split(':')[1].strip())
                results['overall'] = overall_acc
            elif line.startswith('\t') and ':' in line:
                # Parse question type specific accuracies
                parts = line.strip().split(':')
                qtype = parts[0].strip()
                acc_info = parts[1].strip().split('(')
                accuracy = float(acc_info[0].strip())
                count = int(acc_info[1].replace(')', '').strip())
                results[qtype] = {'accuracy': accuracy, 'count': count}
        
        return results, True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {hyp_file}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {}, False

def main():
    # Configuration
    metric_model = 'gpt-4o-mini'  # Default model, can be changed
    ref_file = '/home/samer/Documents/LAU/Research/focus_memgpt/Focused-MemGPT/paper_experiments/longmemeval/data/longmemeval_oracle.json'#'data/longmemeval_oracle.json'
    
    # Check if reference file exists
    if not os.path.exists(ref_file):
        print(f"Reference file {ref_file} not found!")
        return
    
    # Generate beta values from 0.0 to 1.0
    beta_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
    
    # Results storage
    all_results = {}
    summary_data = []
    
    print("="*60)
    print("Running MemGPT Beta Evaluation Across All Beta Values")
    print("="*60)
    
    for beta in beta_values:
        hyp_file = f"memgpt_hypotheses_hybrid_beta{beta}.jsonl"
        
        if not os.path.exists(hyp_file):
            print(f"Warning: {hyp_file} not found, skipping...")
            continue
        
        print(f"\n--- Evaluating Beta {beta} ---")
        
        results, success = run_evaluation(metric_model, hyp_file, ref_file)
        
        if success and results:
            all_results[beta] = results
            
            # Add to summary
            summary_entry = {
                'beta': beta,
                'overall_accuracy': results.get('overall', 0.0)
            }
            
            # Add question type specific accuracies
            for qtype, qdata in results.items():
                if qtype != 'overall' and isinstance(qdata, dict):
                    summary_entry[f'{qtype}_accuracy'] = qdata['accuracy']
                    summary_entry[f'{qtype}_count'] = qdata['count']
            
            summary_data.append(summary_entry)
            
            print(f"Beta {beta}: Overall Accuracy = {results.get('overall', 0.0):.4f}")
        else:
            print(f"Failed to evaluate beta {beta}")
    
    # Save comprehensive results
    results_file = 'beta_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create and save summary DataFrame
    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.sort_values('beta')
        
        # Save as CSV
        csv_file = 'beta_evaluation_summary.csv'
        df.to_csv(csv_file, index=False)
        
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        
        # Print overall accuracy comparison
        print("\nOverall Accuracy by Beta Value:")
        print("-" * 40)
        for _, row in df.iterrows():
            print(f"Beta {row['beta']:.1f}: {row['overall_accuracy']:.4f}")
        
        # Find best beta
        best_beta = df.loc[df['overall_accuracy'].idxmax()]
        print(f"\nBest performing beta: {best_beta['beta']:.1f} (Accuracy: {best_beta['overall_accuracy']:.4f})")
        
        # Print detailed results table
        print(f"\nDetailed Results Table:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        print(f"\nResults saved to:")
        print(f"  - Detailed results: {results_file}")
        print(f"  - Summary CSV: {csv_file}")
        
        # Create a simple plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['beta'], df['overall_accuracy'], 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Beta Value')
            plt.ylabel('Overall Accuracy')
            plt.title('MemGPT Performance Across Beta Values')
            plt.grid(True, alpha=0.3)
            plt.xticks(df['beta'])
            
            # Highlight best performance
            best_idx = df['overall_accuracy'].idxmax()
            plt.plot(df.iloc[best_idx]['beta'], df.iloc[best_idx]['overall_accuracy'], 
                    'ro', markersize=12, label=f'Best: β={df.iloc[best_idx]["beta"]:.1f}')
            plt.legend()
            
            plot_file = 'beta_evaluation_plot.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - Plot: {plot_file}")
            
        except ImportError:
            print("  (Install matplotlib for visualization plot)")
    
    else:
        print("No successful evaluations completed.")

if __name__ == '__main__':
    # Allow override of metric model from command line
    if len(sys.argv) > 1:
        metric_model = sys.argv[1]
        print(f"Using metric model: {metric_model}")
    
    main() 