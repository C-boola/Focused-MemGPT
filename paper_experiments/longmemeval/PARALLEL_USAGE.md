# Parallel LongMemEval Runner Usage Guide

## Overview
The `run_longmemeval_parallel.py` script allows you to run multiple instances of the LongMemEval benchmark in parallel with different beta values. This is particularly useful for testing the hybrid memory mode across multiple beta parameters simultaneously.

## Basic Usage

### Run with default settings (beta 0.0 to 1.0 in 0.1 increments)
```bash
python run_longmemeval_parallel.py
```

### Run in test mode (first 3 cases only)
```bash
python run_longmemeval_parallel.py --test
```

### Specify memory mode
```bash
python run_longmemeval_parallel.py --mode hybrid
python run_longmemeval_parallel.py --mode focus
python run_longmemeval_parallel.py --mode fifo
```

### Custom beta range
```bash
# Run beta from 0.2 to 0.8 in 0.2 increments
python run_longmemeval_parallel.py --beta-start 0.2 --beta-end 0.8 --beta-step 0.2

# Run only specific beta values around 0.5
python run_longmemeval_parallel.py --beta-start 0.4 --beta-end 0.6 --beta-step 0.1
```

### Control parallelism
```bash
# Limit to 4 parallel workers
python run_longmemeval_parallel.py --max-workers 4

# Use only 2 parallel workers for resource-constrained systems
python run_longmemeval_parallel.py --max-workers 2
```

## Common Use Cases

### 1. Full Beta Sweep for Hybrid Mode
```bash
python run_longmemeval_parallel.py --mode hybrid
```
This will run 11 parallel instances (beta 0.0, 0.1, 0.2, ..., 1.0) in hybrid mode.

### 2. Test Run Before Full Benchmark
```bash
python run_longmemeval_parallel.py --mode hybrid --test --max-workers 2
```
This runs a quick test with the first 3 cases only, using 2 parallel workers.

### 3. Fine-grained Beta Testing
```bash
python run_longmemeval_parallel.py --mode hybrid --beta-start 0.45 --beta-end 0.55 --beta-step 0.05
```
This tests beta values: 0.45, 0.5, 0.55 (useful for fine-tuning around optimal values).

## Output Files
Each beta value generates its own output file:
- `memgpt_hypotheses_hybrid_beta0.0.jsonl`
- `memgpt_hypotheses_hybrid_beta0.1.jsonl`
- ...
- `memgpt_hypotheses_hybrid_beta1.0.jsonl`

For beta=0.5, the file is named `memgpt_hypotheses_hybrid.jsonl` (no beta suffix).

## Resource Considerations
- Each parallel worker runs a full MemGPT agent instance
- Memory usage scales with number of workers
- Consider using `--max-workers` to limit resource usage
- Monitor system resources during execution

## Resume Support
Each individual run supports resume functionality from the original script. If a parallel run is interrupted, you can restart the same command and completed beta values will be skipped.

## Real-time Progress Monitoring
The script provides clean, organized progress monitoring:

### 1. **Filtered Output**: Only important messages are shown
- `[PROGRESS β=X.X]` - Progress bars and completion percentages
- `[START β=X.X]` - When each beta value begins
- `[MEMORY β=X.X]` - Memory operations (summarization, token reduction)
- `[ERROR β=X.X]` - Any errors or failures
- `[COMPLETE β=X.X]` - When each beta value finishes

### 2. **Consolidated Progress Summary** (every 30 seconds)
```
================================================================================
                           CONSOLIDATED PROGRESS SUMMARY                           
================================================================================
β=0.0 │ ████████░░░░░░░░░░░░░░░░░░░░░░ │  26.6% │  77/289 │ ETA: 15:23
β=0.1 │ ██████░░░░░░░░░░░░░░░░░░░░░░░░ │  20.4% │  59/289 │ ETA: 18:42
β=0.2 │ ███████░░░░░░░░░░░░░░░░░░░░░░░ │  23.2% │  67/289 │ ETA: 16:15
β=0.3 │ ████████░░░░░░░░░░░░░░░░░░░░░░ │  26.6% │  77/289 │ ETA: 15:23
β=0.4 │ █████░░░░░░░░░░░░░░░░░░░░░░░░░ │  17.3% │  50/289 │ ETA: 19:45
β=0.5 │ ██████████░░░░░░░░░░░░░░░░░░░░ │  33.2% │  96/289 │ ETA: 12:18
================================================================================
```

### 3. **Individual Progress Lines**
```
[PROGRESS β=0.5] 33%|███▍      | 96/289 [12:18<25:32,  7.95s/it]
[MEMORY β=0.3] HYBRID summarization triggered
[MEMORY β=0.3] Memory freed: 84391 tokens
[PROGRESS β=0.3] 27%|██▋       | 77/289 [15:23<41:22, 11.71s/it]
```

## Error Handling
- Failed runs are reported in the summary
- Each beta value runs independently
- One failed beta doesn't affect others
- Check output files to verify successful completion 