#!/usr/bin/env python3
"""
Parallel runner for LongMemEval benchmark with multiple beta values.
This script spawns multiple processes to run the original run_longmemeval.py
with different beta values in parallel.
"""

import subprocess
import sys
import os
import time
import argparse
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict

# Global progress tracking
progress_tracker = {}
progress_lock = threading.Lock()

def update_progress(beta: float, current: int, total: int, eta: str = ""):
    """Update progress for a specific beta value."""
    with progress_lock:
        progress_tracker[beta] = {
            'current': current,
            'total': total,
            'eta': eta,
            'percentage': (current / total * 100) if total > 0 else 0
        }

def print_progress_summary():
    """Print a consolidated progress summary for all beta values."""
    with progress_lock:
        if not progress_tracker:
            return
        
        print(f"\n{'='*80}")
        print(f"{'CONSOLIDATED PROGRESS SUMMARY':^80}")
        print(f"{'='*80}")
        
        for beta in sorted(progress_tracker.keys()):
            data = progress_tracker[beta]
            percentage = data['percentage']
            current = data['current']
            total = data['total']
            eta = data['eta']
            
            # Create a simple progress bar
            bar_length = 30
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"β={beta:3.1f} │ {bar} │ {percentage:5.1f}% │ {current:3d}/{total:3d} │ ETA: {eta}")
        
        print(f"{'='*80}\n")

def run_single_beta(beta: float, memory_mode: str = "hybrid", test_mode: bool = False) -> Tuple[float, int, str]:
    """
    Run a single beta value using the original script with real-time output streaming.
    
    Args:
        beta: Beta value to run
        memory_mode: Memory mode to use (focus, fifo, hybrid)
        test_mode: Whether to run in test mode
        
    Returns:
        Tuple of (beta, return_code, output)
    """
    script_path = os.path.join(os.path.dirname(__file__), "run_longmemeval.py")
    
    # Build command
    cmd = [sys.executable, script_path, "--mode", memory_mode, "--beta", str(beta)]
    if test_mode:
        cmd.append("--test")
    
    print(f"[β={beta}] Starting: {' '.join(cmd)}")
    
    try:
        # Start the subprocess with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        output_lines = []
        
        # Read output line by line in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.rstrip()
                output_lines.append(line)
                
                # Filter and format important messages only
                should_print = False
                formatted_line = ""
                
                # Color code different beta values for better visibility
                colors = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
                color = colors[int(beta * 10) % len(colors)]
                reset_color = '\033[0m'
                bold = '\033[1m'
                
                # Progress bars - make them very prominent
                if "Overall Progress:" in line:
                    progress_part = line.split("Overall Progress:")[1].strip()
                    formatted_line = f"{color}{bold}[PROGRESS β={beta}]{reset_color} {progress_part}"
                    should_print = True
                    
                    # Extract progress information for the tracker
                    try:
                        # Parse something like "15%|██▍     | 23/150 [02:45<15:23,  7.28s/it]"
                        if "|" in progress_part and "/" in progress_part:
                            # Extract current/total from "23/150"
                            fraction_part = progress_part.split("|")[2].strip().split()[0]
                            if "/" in fraction_part:
                                current_str, total_str = fraction_part.split("/")
                                current = int(current_str)
                                total = int(total_str)
                                
                                # Extract ETA if available
                                eta = ""
                                if "<" in progress_part:
                                    eta_part = progress_part.split("<")[1].split(",")[0]
                                    eta = eta_part
                                
                                update_progress(beta, current, total, eta)
                    except (ValueError, IndexError):
                        pass  # Ignore parsing errors
                
                # Starting/completion messages
                elif "MemGPT LongMemEval Benchmark Script" in line:
                    formatted_line = f"{color}[START β={beta}]{reset_color} Benchmark Starting"
                    should_print = True
                elif "BENCHMARK RUN COMPLETE" in line or "TEST RUN COMPLETE" in line:
                    formatted_line = f"{color}{bold}[COMPLETE β={beta}]{reset_color} Benchmark Finished!"
                    should_print = True
                
                # Memory operations - important for understanding performance
                elif "MANUAL" in line and "SUMMARIZATION TRIGGERED" in line:
                    mode = "FOCUS" if "FOCUS" in line else "HYBRID" if "HYBRID" in line else "FIFO"
                    formatted_line = f"{color}[MEMORY β={beta}]{reset_color} {mode} summarization triggered"
                    should_print = True
                elif "After summarization:" in line and "reduced by" in line:
                    # Extract the reduction info
                    parts = line.split("reduced by")
                    if len(parts) > 1:
                        reduction = parts[1].strip().rstrip(")")
                        formatted_line = f"{color}[MEMORY β={beta}]{reset_color} Memory freed: {reduction} tokens"
                        should_print = True
                
                # Error messages
                elif "ERROR" in line or "FAILED" in line or "Exception" in line:
                    formatted_line = f"{color}[ERROR β={beta}]{reset_color} {line}"
                    should_print = True
                
                # Starting test instances - but only every 10th one to reduce noise
                elif "Starting Test Instance:" in line:
                    # Extract instance number from the progress context
                    instance_info = line.split("Starting Test Instance:")[1].strip()
                    formatted_line = f"{color}[INSTANCE β={beta}]{reset_color} {instance_info}"
                    # Only print every 10th instance to reduce noise
                    if "(" in line:  # Has test number info
                        should_print = True  # For now, print all - can adjust if still too noisy
                
                # Print the formatted line if it's important
                if should_print:
                    print(formatted_line)
        
        # Wait for the process to complete
        return_code = process.wait()
        full_output = '\n'.join(output_lines)
        
        if return_code == 0:
            print(f"[β={beta}] ✓ COMPLETED SUCCESSFULLY")
        else:
            print(f"[β={beta}] ✗ FAILED with return code {return_code}")
        
        return beta, return_code, full_output
        
    except subprocess.TimeoutExpired:
        print(f"[β={beta}] ✗ TIMED OUT")
        return beta, -1, "TIMEOUT"
    except Exception as e:
        print(f"[β={beta}] ✗ EXCEPTION: {e}")
        return beta, -2, str(e)

def main():
    """Main function to orchestrate parallel runs."""
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark in parallel with multiple beta values")
    parser.add_argument("--mode", choices=["focus", "fifo", "hybrid"], default="hybrid",
                       help="Memory mode to use (default: hybrid)")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (first 3 cases only)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: number of CPU cores)")
    parser.add_argument("--beta-start", type=float, default=0.0,
                       help="Starting beta value (default: 0.0)")
    parser.add_argument("--beta-end", type=float, default=1.0,
                       help="Ending beta value (default: 1.0)")
    parser.add_argument("--beta-step", type=float, default=0.1,
                       help="Beta increment step (default: 0.1)")
    
    args = parser.parse_args()
    
    # Generate beta values
    beta_values = []
    current_beta = args.beta_start
    while current_beta <= args.beta_end + 1e-9:  # Add small epsilon for floating point comparison
        beta_values.append(round(current_beta, 1))  # Round to avoid floating point precision issues
        current_beta += args.beta_step
    
    print("=" * 70)
    print("  MEMGPT LONGMEMEVAL PARALLEL BENCHMARK RUNNER")
    print("=" * 70)
    print(f"Memory mode: {args.mode}")
    print(f"Test mode: {'ON' if args.test else 'OFF'}")
    print(f"Beta values: {beta_values}")
    print(f"Total runs: {len(beta_values)}")
    print(f"Max workers: {args.max_workers if args.max_workers else 'CPU count'}")
    
    if args.mode != "hybrid":
        print(f"\nWARNING: Beta parameter only affects hybrid mode. Running {args.mode} mode with multiple beta values.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("\nStarting parallel execution...")
    print("=" * 70)
    
    # Initialize progress tracker for all beta values
    for beta in beta_values:
        update_progress(beta, 0, 289 if not args.test else 3, "calculating...")
    
    start_time = time.time()
    
    # Start periodic progress summary thread
    summary_stop_event = threading.Event()
    def periodic_summary():
        while not summary_stop_event.is_set():
            time.sleep(30)  # Print summary every 30 seconds
            if not summary_stop_event.is_set():
                print_progress_summary()
    
    summary_thread = threading.Thread(target=periodic_summary, daemon=True)
    summary_thread.start()
    
    # Use ProcessPoolExecutor for true parallelism
    max_workers = args.max_workers if args.max_workers else min(len(beta_values), os.cpu_count())
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_beta = {
                executor.submit(run_single_beta, beta, args.mode, args.test): beta 
                for beta in beta_values
            }
            
            # Track results
            completed = 0
            failed = 0
            results = []
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_beta):
                beta = future_to_beta[future]
                try:
                    beta_result, return_code, output = future.result()
                    results.append((beta_result, return_code, output))
                    
                    if return_code == 0:
                        completed += 1
                        status = "✓"
                    else:
                        failed += 1
                        status = "✗"
                    
                    progress = completed + failed
                    print(f"\n{'='*50}")
                    print(f"SUMMARY [{progress:2d}/{len(beta_values)}] {status} Beta={beta} | Success: {completed} | Failed: {failed}")
                    print(f"{'='*50}\n")
                    
                except Exception as exc:
                    failed += 1
                    progress = completed + failed
                    print(f"\n{'='*50}")
                    print(f"SUMMARY [{progress:2d}/{len(beta_values)}] ✗ Beta={beta} EXCEPTION: {exc}")
                    print(f"{'='*50}\n")
                    results.append((beta, -3, str(exc)))
    
    finally:
        # Stop the summary thread
        summary_stop_event.set()
    
    # Print final progress summary
    print_progress_summary()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Final summary
    print("\n" + "=" * 70)
    print("  PARALLEL EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Successful runs: {completed}/{len(beta_values)}")
    print(f"Failed runs: {failed}/{len(beta_values)}")
    
    if failed > 0:
        print(f"\nFailed beta values:")
        for beta_result, return_code, output in results:
            if return_code != 0:
                print(f"  - Beta={beta_result}: Return code {return_code}")
    
    # List output files
    script_dir = os.path.dirname(__file__)
    output_files = []
    for beta in beta_values:
        if args.mode == "hybrid" and beta != 0.5:
            filename = f"memgpt_hypotheses_{args.mode}_beta{beta}.jsonl"
        else:
            filename = f"memgpt_hypotheses_{args.mode}.jsonl"
        
        if args.test:
            filename += ".test"
        
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            output_files.append(filename)
    
    if output_files:
        print(f"\nOutput files generated:")
        for filename in sorted(set(output_files)):
            print(f"  - {filename}")
    
    print(f"\nAll parallel runs completed!")

if __name__ == "__main__":
    main() 