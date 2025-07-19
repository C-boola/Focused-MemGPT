#!/usr/bin/env python3
"""
Quick accuracy checker for LongMemEval evaluation results.
Simple script to check accuracy of any evaluation result file.
"""

import json
import os
import sys

def calculate_accuracy(filepath):
    """Calculate accuracy from an evaluation results file."""
    try:
        with open(filepath, 'r') as f:
            results = [json.loads(line.strip()) for line in f if line.strip()]
        
        if not results:
            return None, "No results found in file"
        
        correct = sum(1 for result in results if result.get('autoeval_label', {}).get('label', False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'total_questions': total,
            'correct_answers': correct,
            'incorrect_answers': total - correct,
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100
        }, None
    
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def show_sample_results(filepath, num_samples=3):
    """Show sample results from the file."""
    try:
        with open(filepath, 'r') as f:
            results = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"\n📋 Sample Results:")
        print("-" * 50)
        
        for i, result in enumerate(results[:num_samples]):
            label = result.get('autoeval_label', {}).get('label', False)
            status = "✅ CORRECT" if label else "❌ INCORRECT"
            
            print(f"\n{i+1}. {status}")
            print(f"   Question ID: {result.get('question_id', 'N/A')}")
            print(f"   Ground Truth: {result.get('ground_truth', 'N/A')}")
            hypothesis = result.get('hypothesis', 'N/A')
            if len(hypothesis) > 80:
                hypothesis = hypothesis[:80] + "..."
            print(f"   Hypothesis: {hypothesis}")
        
        if len(results) > num_samples:
            print(f"\n... and {len(results) - num_samples} more results")
    
    except Exception as e:
        print(f"Error showing sample results: {e}")

def find_eval_files():
    """Find all evaluation result files."""
    eval_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.eval-results-gpt-4o-mini') or file.endswith('.eval-results-gpt-4o'):
                eval_files.append(os.path.join(root, file))
    return sorted(eval_files)

def main():
    print("🔍 LongMemEval Quick Accuracy Checker")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # File specified as command line argument
        filepath = sys.argv[1]
        
        if not os.path.exists(filepath):
            print(f"❌ Error: File '{filepath}' not found")
            return
        
        result, error = calculate_accuracy(filepath)
        
        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"📊 Analysis for: {os.path.basename(filepath)}")
            print("-" * 50)
            print(f"Total Questions: {result['total_questions']}")
            print(f"Correct Answers: {result['correct_answers']}")
            print(f"Incorrect Answers: {result['incorrect_answers']}")
            print(f"Accuracy: {result['accuracy']:.4f} ({result['accuracy_percentage']:.2f}%)")
            
            show_sample_results(filepath)
    else:
        # Interactive mode - show available files
        eval_files = find_eval_files()
        
        if not eval_files:
            print("❌ No evaluation result files found in current directory and subdirectories.")
            print("Make sure you're in the correct directory and files end with '.eval-results-gpt-4o-mini'")
            return
        
        print(f"Found {len(eval_files)} evaluation files:")
        print("-" * 50)
        
        for i, filepath in enumerate(eval_files):
            print(f"{i+1:2d}. {os.path.basename(filepath)}")
        
        try:
            choice = input(f"\nEnter file number (1-{len(eval_files)}) or press Enter to analyze all: ").strip()
            
            if choice == "":
                # Analyze all files
                print(f"\n📈 Analyzing all {len(eval_files)} files:")
                print("=" * 60)
                
                for filepath in eval_files:
                    result, error = calculate_accuracy(filepath)
                    if error:
                        print(f"❌ {os.path.basename(filepath)}: {error}")
                    else:
                        print(f"📊 {os.path.basename(filepath):50s} | Accuracy: {result['accuracy_percentage']:5.1f}% ({result['correct_answers']}/{result['total_questions']})")
            
            else:
                file_num = int(choice)
                if 1 <= file_num <= len(eval_files):
                    filepath = eval_files[file_num - 1]
                    result, error = calculate_accuracy(filepath)
                    
                    if error:
                        print(f"❌ Error: {error}")
                    else:
                        print(f"\n📊 Detailed Analysis:")
                        print(f"📁 {os.path.basename(filepath)}")
                        print("-" * 50)
                        print(f"Total Questions: {result['total_questions']}")
                        print(f"Correct Answers: {result['correct_answers']}")
                        print(f"Incorrect Answers: {result['incorrect_answers']}")
                        print(f"Accuracy: {result['accuracy']:.4f} ({result['accuracy_percentage']:.2f}%)")
                        
                        show_sample_results(filepath)
                else:
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(eval_files)}")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main() 