#!/usr/bin/env python3
"""
Batched runner for LongMemEval benchmark to avoid PostgreSQL connection limits.
This script runs beta values in smaller batches to prevent "too many clients" errors.
"""

import subprocess
import sys
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

def run_single_beta(beta: float, memory_mode: str = "hybrid", test_mode: bool = False, cluster_summaries: bool = False) -> Tuple[float, int, str]:
    """Run a single beta value using the original script."""
    script_path = os.path.join(os.path.dirname(__file__), "run_longmemeval.py")
    
    # Build command
    cmd = [sys.executable, script_path, "--mode", memory_mode, "--beta", str(beta)]
    if test_mode:
        cmd.append("--test")
    if cluster_summaries:
        cmd.append("--cluster")
    
    clustering_status = "with clustering" if cluster_summaries else "without clustering"
    print(f"[BATCH] Starting beta={beta} {clustering_status}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        
        if result.returncode == 0:
            print(f"[BATCH] ✓ Beta={beta} completed successfully")
        else:
            print(f"[BATCH] ✗ Beta={beta} failed with return code {result.returncode}")
            if result.stderr:
                print(f"[BATCH] Error: {result.stderr[:200]}...")
        
        return beta, result.returncode, result.stdout
        
    except Exception as e:
        print(f"[BATCH] ✗ Beta={beta} exception: {e}")
        return beta, -2, str(e)

def run_beta_batch(beta_batch: List[float], memory_mode: str, test_mode: bool, cluster_summaries: bool = False) -> List[Tuple[float, int, str]]:
    """Run a batch of beta values in parallel."""
    print(f"\n{'='*60}")
    print(f"RUNNING BATCH: {beta_batch}")
    print(f"{'='*60}")
    
    batch_results = []
    max_workers = min(len(beta_batch), 3)  # Limit to 3 workers per batch
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_beta = {
            executor.submit(run_single_beta, beta, memory_mode, test_mode, cluster_summaries): beta 
            for beta in beta_batch
        }
        
        for future in as_completed(future_to_beta):
            beta = future_to_beta[future]
            try:
                result = future.result()
                batch_results.append(result)
            except Exception as exc:
                print(f"[BATCH] ✗ Beta={beta} generated exception: {exc}")
                batch_results.append((beta, -3, str(exc)))
    
    return batch_results

def main():
    """Main function to run beta values in batches."""
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark in batches to avoid DB connection limits")
    parser.add_argument("--mode", choices=["focus", "fifo", "hybrid"], default="hybrid",
                       help="Memory mode to use (default: hybrid)")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (first 3 cases only)")
    parser.add_argument("--batch-size", type=int, default=3,
                       help="Number of beta values to run in parallel per batch (default: 3)")
    parser.add_argument("--beta-start", type=float, default=0.0,
                       help="Starting beta value (default: 0.0)")
    parser.add_argument("--beta-end", type=float, default=1.0,
                       help="Ending beta value (default: 1.0)")
    parser.add_argument("--beta-step", type=float, default=0.1,
                       help="Beta increment step (default: 0.1)")
    parser.add_argument("--cluster", action="store_true",
                       help="Enable clustering-based summarization (default: disabled)")
    
    args = parser.parse_args()
    
    # Generate beta values
    beta_values = []
    current_beta = args.beta_start
    while current_beta <= args.beta_end + 1e-9:
        beta_values.append(round(current_beta, 1))
        current_beta += args.beta_step
    
    # Split into batches
    batches = []
    for i in range(0, len(beta_values), args.batch_size):
        batch = beta_values[i:i + args.batch_size]
        batches.append(batch)
    
    print("=" * 70)
    print("  MEMGPT LONGMEMEVAL BATCHED BENCHMARK RUNNER")
    print("=" * 70)
    print(f"Memory mode: {args.mode}")
    print(f"Test mode: {'ON' if args.test else 'OFF'}")
    print(f"Clustering-based summarization: {'ENABLED' if args.cluster else 'DISABLED'}")
    print(f"Beta values: {beta_values}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(batches)}")
    print(f"Batches: {batches}")
    
    start_time = time.time()
    all_results = []
    completed = 0
    failed = 0
    
    # Run each batch sequentially
    for batch_num, beta_batch in enumerate(batches, 1):
        print(f"\n🚀 Starting Batch {batch_num}/{len(batches)}")
        batch_results = run_beta_batch(beta_batch, args.mode, args.test, args.cluster)
        all_results.extend(batch_results)
        
        # Update counters
        for beta, return_code, output in batch_results:
            if return_code == 0:
                completed += 1
            else:
                failed += 1
        
        print(f"✅ Batch {batch_num} completed. Overall: {completed} success, {failed} failed")
        
        # Wait between batches to let DB connections close
        if batch_num < len(batches):
            print("⏳ Waiting 10 seconds before next batch...")
            time.sleep(10)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Final summary
    print("\n" + "=" * 70)
    print("  BATCHED EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Successful runs: {completed}/{len(beta_values)}")
    print(f"Failed runs: {failed}/{len(beta_values)}")
    
    if failed > 0:
        print(f"\nFailed beta values:")
        for beta, return_code, output in all_results:
            if return_code != 0:
                print(f"  - Beta={beta}: Return code {return_code}")
    
    # List output files
    script_dir = os.path.dirname(__file__)
    output_files = []
    for beta in beta_values:
        output_filename = f"memgpt_hypotheses_prompted_{args.mode}"
        if args.mode == "hybrid":
            output_filename += f"_beta{beta}"
        if args.cluster:
            output_filename += "_cluster"
        filename = f"{output_filename}.jsonl"
        
        if args.test:
            filename += ".test"
        
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            output_files.append(filename)
    
    if output_files:
        print(f"\nOutput files generated:")
        for filename in sorted(set(output_files)):
            print(f"  - {filename}")
    
    print(f"\nAll batched runs completed!")

if __name__ == "__main__":
    main() 