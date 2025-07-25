#!/bin/bash

# Sequential experiment runner for LongMemEval benchmark
# This script runs multiple configurations one after the other

echo "===== Starting Sequential Experiment Runs ====="
echo "Total experiments to run: 12"
echo "Start time: $(date)"
echo ""

# Function to run a command and check if it succeeded
run_experiment() {
    local cmd="$1"
    local exp_num="$2"
    local total="$3"
    
    echo "===== Experiment $exp_num/$total ====="
    echo "Command: $cmd"
    echo "Starting at: $(date)"
    echo ""
    
    # Run the command
    eval "$cmd"
    local exit_code=$?
    
    echo ""
    echo "Experiment $exp_num completed at: $(date)"
    if [ $exit_code -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED (exit code: $exit_code)"
        echo "ERROR: Experiment $exp_num failed. Do you want to continue? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping execution due to failure."
            exit $exit_code
        fi
    fi
    echo "----------------------------------------"
    echo ""
}

# Array of commands to run
commands=(
    "python run_longmemeval_cases.py --mode hybrid --beta 0.0 --batch-size 10 --trunc-frac 0.5 --cluster False --test"
    "python run_longmemeval_cases.py --mode hybrid --beta 1.0 --batch-size 10 --trunc-frac 0.5 --cluster False"
    "python run_longmemeval_cases.py --mode hybrid --beta 0.0 --batch-size 10 --trunc-frac 0.25 --cluster False"
    "python run_longmemeval_cases.py --mode hybrid --beta 1.0 --batch-size 10 --trunc-frac 0.25 --cluster False"
    "python run_longmemeval_cases.py --mode density --beta 1.0 --batch-size 10 --cluster True"
    "python run_longmemeval_cases.py --mode pure_cluster --batch-size 10 --cluster True"
    "python run_longmemeval_cases.py --mode hybrid --beta 0.0 --batch-size 10 --trunc-frac 0.5 --cluster True"
    "python run_longmemeval_cases.py --mode hybrid --beta 1.0 --batch-size 10 --trunc-frac 0.5 --cluster True"
    
    # "python run_longmemeval_cases.py --mode density --beta 0.0 --batch-size 10 --cluster"
)

# Run each experiment
total_experiments=${#commands[@]}
for i in "${!commands[@]}"; do
    exp_num=$((i + 1))
    run_experiment "${commands[$i]}" "$exp_num" "$total_experiments"
done

echo "===== All Experiments Complete ====="
echo "End time: $(date)"
echo "All $total_experiments experiments have been executed." 