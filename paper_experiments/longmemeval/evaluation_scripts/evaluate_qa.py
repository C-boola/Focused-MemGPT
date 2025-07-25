import os
import sys
import json
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dotenv import load_dotenv

load_dotenv()


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


def create_metric_client(metric_model_source, openai_api_key, openai_api_base):
    """Create OpenAI client for metric evaluation"""
    return OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )


def evaluate_single_entry(entry_data):
    """
    Evaluate a single hypothesis entry.
    entry_data is a tuple containing (entry, qid2qdata, qid2qtype, metric_model, metric_config, verbose)
    """
    entry, qid2qdata, qid2qtype, metric_model, metric_config, verbose = entry_data
    
    # Create client for this process
    metric_client = create_metric_client(
        metric_config['source'], 
        metric_config['api_key'], 
        metric_config['api_base']
    )
    
    if entry['question_id'] not in qid2qtype:
        return None, f"Warning: skipping {entry['question_id']} as it is not in reference data."
    
    qtype = qid2qtype[entry['question_id']]
    q = qid2qdata[entry['question_id']]['question']
    ans = qid2qdata[entry['question_id']]['answer']
    hyp = entry['hypothesis']
    
    prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
    kwargs = {
        'model': metric_model,
        'messages':[
            {"role": "user", "content": prompt}
        ],
        'n': 1,
        'temperature': 0,
        'max_tokens': 10
    }
    
    try:
        completion = chat_completions_with_backoff(metric_client, **kwargs)
        eval_response = completion.choices[0].message.content.strip()
        label = 'yes' in eval_response.lower()
        
        entry['autoeval_label'] = {
            'model': metric_model,
            'label': label
        }
        entry['question_type'] = qtype
        
        if verbose:
            log_entry = {
                'question': q,
                'answer': ans,
                'hypothesis': hyp,
                'autoeval_label': label
            }
            return entry, json.dumps(log_entry, indent=4)
        
        return entry, None
        
    except Exception as e:
        return None, f"Error processing {entry['question_id']}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Evaluate QA results with parallel processing')
    parser.add_argument('metric_model', help='Metric model to use for evaluation')
    parser.add_argument('ref_file', help='Reference file path')
    parser.add_argument('hyp_file', help='Hypothesis file path')
    parser.add_argument('--max-workers', type=int, default=4, 
                       help='Maximum number of worker processes (default: 4)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    metric_model_short = args.metric_model
    ref_file = args.ref_file
    hyp_file = args.hyp_file
    max_workers = args.max_workers
    verbose = args.verbose
    
    result_file = hyp_file + '.eval-results-{}'.format(metric_model_short)

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        sys.exit(1)
        
    metric_model, metric_model_source = model_zoo[metric_model_short]
    
    # Setup API configuration
    if metric_model_source == 'openai':
        openai.organization = os.getenv('OPENAI_ORGANIZATION')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_api_base = None
    else:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"
    
    metric_config = {
        'source': metric_model_source,
        'api_key': openai_api_key,
        'api_base': openai_api_base
    }

    try:
        hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    except:
        hypotheses = json.load(open(hyp_file))
    try:
        references = json.load(open(ref_file))
    except:
        references = [json.loads(line) for line in open(ref_file).readlines()]
        
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}
    
    # Debug: Show question type distribution in reference file
    ref_qtype_counts = {}
    for qtype in qid2qtype.values():
        ref_qtype_counts[qtype] = ref_qtype_counts.get(qtype, 0) + 1
    print("Question types in reference file:")
    for qtype, count in ref_qtype_counts.items():
        print(f"  {qtype}: {count} questions")
    
    # Debug: Show question IDs in hypothesis file
    hyp_question_ids = [entry['question_id'] for entry in hypotheses]
    print(f"\nHypothesis file contains {len(hyp_question_ids)} entries")
    
    # Debug: Check which hypothesis question IDs have corresponding reference data
    matched_qtypes = {}
    unmatched_qids = []
    for qid in hyp_question_ids:
        if qid in qid2qtype:
            qtype = qid2qtype[qid]
            matched_qtypes[qtype] = matched_qtypes.get(qtype, 0) + 1
        else:
            unmatched_qids.append(qid)
    
    print("Matched question types from hypothesis file:")
    for qtype, count in matched_qtypes.items():
        print(f"  {qtype}: {count} questions")
    
    if unmatched_qids:
        print(f"\nUnmatched question IDs in hypothesis file: {len(unmatched_qids)}")
        print(f"First few unmatched IDs: {unmatched_qids[:5]}")
    print()
    
    print(f"Using {max_workers} worker processes for parallel evaluation...")

    # Prepare data for parallel processing
    entry_data_list = [
        (entry, qid2qdata, qid2qtype, metric_model, metric_config, verbose)
        for entry in hypotheses
    ]

    logs = []
    with open(result_file, 'w') as out_f:
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and track them with tqdm
            results = list(tqdm(
                executor.map(evaluate_single_entry, entry_data_list),
                total=len(entry_data_list),
                desc="Evaluating entries"
            ))
            
            # Process results
            for result, log_output in results:
                if result is None:
                    if log_output:  # This is a warning or error message
                        print(log_output)
                    continue
                
                logs.append(result)
                if log_output and verbose:  # This is verbose output
                    print(log_output, flush=True)
                    
                print(json.dumps(result), file=out_f)
                qtype2acc[result['question_type']].append(
                    1 if result['autoeval_label']['label'] else 0
                )

    print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
    for k,v in qtype2acc.items():
        if len(v) > 0:
            accuracy = round(np.mean(v), 4)
            print('\t{}: {} ({})'.format(k, accuracy, len(v)))
        else:
            print('\t{}: N/A (0) - No questions of this type in hypothesis file'.format(k))

    print('Saved to', result_file)


if __name__ == '__main__':
    main()
