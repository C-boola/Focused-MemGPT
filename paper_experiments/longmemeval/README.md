# MemGPT LongMemEval Benchmark

This benchmark is designed to test MemGPT's long-term memory and context management capabilities using the LongMemEval dataset. It uses a custom script (`run_longmemeval.py`) that directly controls a MemGPT agent to ensure that memory management functions like context overflow, summarization, and embedding generation are explicitly triggered and tested.

## Setup: Downloading the Data

Before running the benchmark, you must download the required dataset files and place them in the `data/` directory.

1.  Navigate to the data directory from the root of this experiment:
    ```bash
    cd ./data
    ```

2.  Download the main data file (`longmemeval_s.json`) and the oracle file (`longmemeval_oracle.json`) from the Hugging Face Hub.
    ```bash
    wget https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json
    wget https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle.json
    ```

3.  Once the downloads are complete, return to the experiment's root directory:
    ```bash
    cd ..
    ```

## How It Works

This benchmark script is a powerful tool for testing MemGPT's core memory features because it bypasses the standard client-server model and directly instantiates and controls the `Agent` class. This allows for a more controlled and observable experimental setup.

Key features of the implementation:

1.  **Direct Agent Control**: Instead of using `memgpt run` or the MemGPT client, the script creates an `Agent` object directly in memory. This provides full control over the agent's lifecycle and avoids issues with persistent state between runs.

2.  **Surgical History Injection**: For each test case, the full conversation history from the dataset is injected directly into the agent's message queue using `agent.append_to_messages()`. This method "primes" the agent's memory without triggering any premature LLM responses.

3.  **Forced Context Overflow**: After injecting the history, the script manually checks if the context window has overflowed. If it has, it explicitly calls the agent's internal summarization logic (`agent.summarize_messages_inplace()`), forcing it to perform either **Focus** or **FIFO** memory management. This is the core of the test, as it directly evaluates the agent's ability to handle large contexts.

4.  **Custom Embedding Generation**: The script includes a custom-built embedding function that works with the plain-text format of the LongMemEval dataset. This was implemented to work around the default MemGPT embedding function which expects JSON-formatted messages.

5.  **Resumption and Interruption Handling**: The script automatically detects and resumes from a partially completed run by checking the `memgpt_hypotheses.jsonl` output file. It also gracefully saves progress if interrupted with `Ctrl+C`.

## How to Run the Benchmark

### Step 1: Navigate to the Experiment Directory

```bash
cd /path/to/Focused-MemGPT/paper_experiments/longmemeval
```

### Step 2: Run the Benchmark

You can run the benchmark script with several command-line options to control its behavior.

**Choosing a Memory Mode (`--mode`)**

The `--mode` flag allows you to specify which memory management strategy to use. If omitted, it defaults to `focus`.

*   **Run with FIFO mode:**
    ```bash
    python run_longmemeval.py --mode fifo
    ```
*   **Run with Focus mode:**
    ```bash
    python run_longmemeval.py --mode focus
    ```

**Using Test Mode (`--test`)**

The `--test` flag runs only the first 3 test cases with verbose debug output, which is highly recommended for debugging and quick verification.

*   **Run a quick test using FIFO mode:**
    ```bash
    python run_longmemeval.py --test --mode fifo
    ```
*   **Run a quick test using the default Focus mode:**
    ```bash
    python run_longmemeval.py --test
    ```

### Step 3: Generated Output

The script will generate a file named `memgpt_hypotheses.jsonl`. Each line in this file corresponds to a test case and contains the agent's generated answer (the "hypothesis").

## How to Evaluate the Results

After the benchmark run is complete, you can use the provided evaluation scripts to score the agent's performance. This is a two-step process.

### Step 1: Judge the Agent's Answers (`evaluate_qa.py`)

This script uses a powerful "judge" model (e.g., `gpt-4o-mini`) to compare the agent's generated answers (`memgpt_hypotheses.jsonl`) with the ground-truth answers (`data/longmemeval_oracle.json`). It creates a new file containing these judgments.

**Prerequisite: Set Your API Key**
For this step to work, you must provide an OpenAI API key. The most reliable method is to prepend it to the command.

**Run the Evaluation**
Use the following command, replacing `sk-...` with your key. The script requires three arguments: the metric model, the hypothesis file, and the reference file.

```bash
OPENAI_API_KEY="sk-..." python evaluation_scripts/evaluate_qa.py gpt-4o-mini memgpt_hypotheses.jsonl data/longmemeval_oracle.json
```

This command will create a new results file, such as `memgpt_hypotheses.jsonl.eval-results-gpt-4o-mini`.

### Step 2: Print the Final Metrics (`print_qa_metrics.py`)

This script takes the results from the previous step and prints a clean, final report of the accuracy, broken down by task. It requires two arguments: the evaluation results file from Step 1 and the original reference file.

**Run the Metrics Script**
```bash
python evaluation_scripts/print_qa_metrics.py memgpt_hypotheses.jsonl.eval-results-gpt-4o-mini data/longmemeval_oracle.json
```

This will display the final accuracy scores in your terminal.

## How to Test Your Own MemGPT Modifications

This benchmark is an ideal environment for testing changes to MemGPT's core memory management logic.

Because `run_longmemeval.py` directly imports and uses the `Agent` class and its methods, any modifications you make to the underlying MemGPT library files (e.g., to `memgpt/agent.py` or `memgpt/memory.py`) will be automatically used when you run the benchmark.

**Example Workflow:**

1.  Modify the FIFO or Focus mode logic in `memgpt/agent.py`.
2.  Run the benchmark in test mode: `python run_longmemeval.py --test`.
3.  Observe the detailed log output to see if your changes are behaving as expected.
4.  Once you are satisfied, run the full benchmark and use the evaluation scripts to see how your changes impacted performance. 