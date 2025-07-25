# Beta Evaluation Script

This script automatically evaluates MemGPT hypothesis files across different beta values (0.0 to 1.0) and provides comprehensive results.

## Files Created

- `run_beta_evaluation.py` - Main evaluation script
- `requirements_beta_eval.txt` - Python dependencies
- `BETA_EVALUATION_README.md` - This documentation

## Usage

### Basic Usage
```bash
cd Focused-MemGPT/paper_experiments/longmemeval
python run_beta_evaluation.py
```

### With Custom Metric Model
```bash
python run_beta_evaluation.py gpt-4o
```

Available metric models (from the original script):
- `gpt-4o-mini` (default)
- `gpt-4o`
- `llama-3.1-70b-instruct`

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements_beta_eval.txt
```

2. Ensure you have:
   - All hypothesis files: `memgpt_hypotheses_hybrid_beta{X}.jsonl` where X = 0.0, 0.1, ..., 1.0
   - Reference file: `data/longmemeval_oracle.json`
   - OpenAI API key (if using OpenAI models)

## Output Files

The script generates several output files:

1. **`beta_evaluation_results.json`** - Detailed results for each beta value
2. **`beta_evaluation_summary.csv`** - Summary table with all metrics
3. **`beta_evaluation_plot.png`** - Performance visualization (if matplotlib available)
4. **Individual evaluation files** - `{hypothesis_file}.eval-results-{model}` for each beta

## Expected Output

The script will:
1. Run evaluation on each beta value sequentially
2. Print progress and individual results
3. Display a comprehensive summary table
4. Identify the best performing beta value
5. Save all results in multiple formats

### Sample Console Output
```
============================================================
Running MemGPT Beta Evaluation Across All Beta Values
============================================================

--- Evaluating Beta 0.0 ---
Running evaluation for memgpt_hypotheses_hybrid_beta0.0.jsonl...
Beta 0.0: Overall Accuracy = 0.7234

--- Evaluating Beta 0.1 ---
Running evaluation for memgpt_hypotheses_hybrid_beta0.1.jsonl...
Beta 0.1: Overall Accuracy = 0.7456

[... continues for all beta values ...]

============================================================
SUMMARY RESULTS
============================================================

Overall Accuracy by Beta Value:
----------------------------------------
Beta 0.0: 0.7234
Beta 0.1: 0.7456
[... etc ...]

Best performing beta: 0.5 (Accuracy: 0.8123)

Detailed Results Table:
[Comprehensive table with all metrics]
```

## Troubleshooting

1. **Missing files**: Ensure all hypothesis files exist in the current directory
2. **API errors**: Check OpenAI API key if using OpenAI models
3. **Dependencies**: Install missing packages with pip
4. **Permission errors**: Make sure the script is executable (`chmod +x run_beta_evaluation.py`)

## Notes

- The script automatically detects available hypothesis files
- Results are sorted by beta value for easy comparison
- The script handles missing files gracefully
- All intermediate evaluation files are preserved for detailed analysis 