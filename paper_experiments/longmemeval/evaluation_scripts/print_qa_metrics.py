import json
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python print_qa_metrics.py in_file ref_file')
        exit()

    in_file = sys.argv[1]
    ref_file = sys.argv[2]

    try:
        # Load the evaluation results (JSONL file)
        results = [json.loads(line) for line in open(in_file).readlines()]
        
        # Load the reference data (JSON file)
        references = json.load(open(ref_file))
    except Exception as e:
        print(f"Error: Failed to load or parse input files. Please check file paths and format.")
        print(f"Details: {e}")
        exit()

    # Create a mapping from question_id to question_type for easy lookup
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}

    # Dictionary to hold accuracy scores for each question type
    qtype2acc = {}

    for entry in results:
        label = 1 if entry.get('autoeval_label', {}).get('label') else 0
        
        # Get the question type using the mapping
        q_id = entry.get('question_id')
        if not q_id:
            print(f"Warning: Skipping entry due to missing 'question_id'. Entry: {entry}")
            continue
            
        qtype = qid2qtype.get(q_id)
        if not qtype:
            print(f"Warning: Skipping question_id {q_id} as it's not in the reference file.")
            continue

        if qtype not in qtype2acc:
            qtype2acc[qtype] = []
        qtype2acc[qtype].append(label)

    # Calculate and print overall accuracy
    all_labels = [label for labels in qtype2acc.values() for label in labels]
    if all_labels:
        overall_accuracy = np.mean(all_labels)
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({len(all_labels)} samples)")
    else:
        print("\nNo results to calculate accuracy.")

    # Print accuracy for each task
    print('\nEvaluation results by task:')
    # Get all possible types from the reference file to print `nan` for missing ones
    all_qtypes = sorted(list(set(qid2qtype.values())))
    for qtype in all_qtypes:
        scores = qtype2acc.get(qtype, [])
        if scores:
            accuracy = np.mean(scores)
            print(f"\t{qtype}: {accuracy:.4f} ({len(scores)})")
        else:
            # This handles cases where a question type exists but had no results in the input file
            print(f"\t{qtype}: nan (0)")
