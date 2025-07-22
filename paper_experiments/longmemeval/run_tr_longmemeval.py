#!/usr/bin/env python3
"""
Parallel test case runner for LongMemEval benchmark - TEMPORAL REASONING ONLY.
This script runs only test cases with question_type "temporal-reasoning".
Works in both test and normal modes with configurable batch sizes.
"""

import json
import os
import uuid
from tqdm import tqdm
import time
import sys
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock
import argparse

# --- Debug Logging Utility ---
DEBUG = True

def log_debug(message):
    """A simple, toggleable logging function."""
    if DEBUG:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}")

# --- MemGPT and Benchmark Imports ---
from memgpt.agent import Agent
from memgpt.interface import AgentInterface
from memgpt.data_types import AgentState
from memgpt.presets.presets import available_presets
from memgpt.config import MemGPTConfig
from memgpt.constants import DEFAULT_PRESET, DEFAULT_PERSONA, DEFAULT_HUMAN, MESSAGE_SUMMARY_WARNING_FRAC
from memgpt.utils import get_persona_text, get_human_text, count_tokens
from memgpt.prompts import gpt_system
from memgpt.presets.presets import generate_functions_json

# Silent interface to suppress output during parallel processing
class SilentInterface(AgentInterface):
    def user_message(self, msg, msg_obj=None) -> None: 
        pass  # Suppress all output in parallel mode
    
    def internal_monologue(self, msg, msg_obj=None) -> None: 
        pass  # Suppress all output in parallel mode
    
    def assistant_message(self, msg, msg_obj=None) -> None: 
        pass  # Suppress all output in parallel mode
    
    def function_message(self, msg, msg_obj=None) -> None: 
        pass  # Suppress all output in parallel mode

# --- Constants ---
MODULE_BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_s.json")
ORACLE_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_oracle.json")

# Only temporal reasoning question types
TEMPORAL_REASONING_QUESTION_TYPES = {
    "temporal-reasoning"
}

# Global file locks for thread-safe writing
output_lock = Lock()

def load_and_filter_data() -> list[dict]:
    """Loads and filters LongMemEval data for temporal reasoning question types only."""
    log_debug("Loading and filtering data for temporal reasoning...")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    with open(ORACLE_PATH, 'r', encoding='utf-8') as f:
        oracle_list = json.load(f)
    
    oracle_data = {item['question_id']: item for item in oracle_list}
    
    filtered_data = [
        item for item in full_data
        if oracle_data.get(item['question_id'], {}).get('question_type') in TEMPORAL_REASONING_QUESTION_TYPES
    ]
    
    log_debug(f"Loaded {len(filtered_data)} temporal reasoning test cases (filtered from {len(full_data)} total)")
    
    if not filtered_data:
        print("CRITICAL WARNING: No temporal reasoning test cases were loaded after filtering!")
    
    return filtered_data

def run_single_test_case(args_tuple):
    """
    Worker function to run a single test case.
    Takes a tuple of arguments to work with multiprocessing.
    """
    (test_case, config_dict, memory_mode, beta, cluster_summaries, 
     prompt_type, centroid_method, score_mode, worker_id) = args_tuple
    
    q_id = test_case['question_id']
    
    try:
        # Recreate config object from dict (needed for multiprocessing)
        config = MemGPTConfig(**config_dict)
        
        # Run the test instance
        hypothesis = run_test_instance_worker(
            config, test_case, memory_mode, beta, cluster_summaries,
            prompt_type, centroid_method, score_mode, worker_id
        )
        
        result = {
            "question_id": q_id,
            "hypothesis": hypothesis,
            "ground_truth": test_case.get('answer', 'N/A')
        }
        
        return {
            "success": True,
            "result": result,
            "worker_id": worker_id,
            "question_id": q_id
        }
        
    except Exception as e:
        error_msg = f"ERROR in worker {worker_id} for case {q_id}: {str(e)}"
        
        return {
            "success": False,
            "result": {
                "question_id": q_id,
                "hypothesis": f"ERROR: {error_msg}",
                "ground_truth": test_case.get('answer', 'N/A')
            },
            "worker_id": worker_id,
            "question_id": q_id,
            "error": error_msg
        }

def run_test_instance_worker(base_config: MemGPTConfig, test_case: dict, memory_mode: str = "focus", 
                           beta: float = 0.5, cluster_summaries: bool = False, 
                           prompt_type: str = "memgpt_default", centroid_method: str = "centroid", 
                           score_mode: str = None, worker_id: int = 0) -> str:
    """
    Worker version of run_test_instance that runs in a separate process.
    Streamlined for parallel execution.
    """
    q_id = test_case['question_id']
    agent_name = f"tr_longmemeval_agent_{q_id}_w{worker_id}"
    
    # 1. Create Agent
    dummy_user_id = uuid.uuid4()
    preset_config = available_presets[DEFAULT_PRESET]
    preset_system_prompt = preset_config["system_prompt"]
    preset_function_set_names = preset_config["functions"]
    functions_schema = generate_functions_json(preset_function_set_names)
    
    agent_state = AgentState(
        name=agent_name,
        user_id=dummy_user_id,
        persona=get_persona_text(DEFAULT_PERSONA),
        human=get_human_text(DEFAULT_HUMAN),
        preset=DEFAULT_PRESET,
        llm_config=base_config.default_llm_config,
        embedding_config=base_config.default_embedding_config,
        state={
            "persona": get_persona_text(DEFAULT_PERSONA),
            "human": get_human_text(DEFAULT_HUMAN),
            "system": gpt_system.get_system_text(preset_system_prompt),
            "functions": functions_schema,
            "messages": None,
            "mem_mode": memory_mode,
            "beta": beta,
            "cluster_summaries": cluster_summaries,
            "prompt_type": prompt_type,
            "centroid_method": centroid_method,
            "score_mode": score_mode,
        },
    )

    try:
        agent = Agent(
            interface=SilentInterface(), 
            agent_state=agent_state, 
            mem_mode=memory_mode, 
            beta=beta, 
            cluster_summaries=cluster_summaries, 
            centroid_method=centroid_method, 
            score_mode=score_mode
        )
    except Exception as e:
        return f"ERROR: AGENT_INSTANTIATION_FAILED FOR {q_id}: {e}"
    
    # 2. Inject conversation history
    full_chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    agent.append_to_messages(full_chat_history)

    # 3. Check for context overflow and trigger summarization if needed
    current_tokens = sum(count_tokens(str(msg)) for msg in agent.messages)
    context_window = agent.agent_state.llm_config.context_window
    overflow_threshold = MESSAGE_SUMMARY_WARNING_FRAC * context_window
    
    if current_tokens > overflow_threshold:
        try:
            if memory_mode == "focus":
                from memgpt.embeddings import calculate_centroid, calculate_medoid, calculate_cosine_distances
                import numpy as np
                
                message_pair_embeddings_with_ids = create_custom_message_pair_embeddings(
                    message_sequence=agent._messages, 
                    embedding_config=agent.agent_state.embedding_config
                )
                
                if message_pair_embeddings_with_ids:
                    actual_embedding_vectors = [pair[2] for pair in message_pair_embeddings_with_ids]
                    if centroid_method == "medoid":
                        centroid_vec = calculate_medoid(actual_embedding_vectors)
                    else:
                        centroid_vec = calculate_centroid(actual_embedding_vectors)
                    
                    if centroid_vec is not None:
                        np_embedding_vectors = [np.array(vec) for vec in actual_embedding_vectors]
                        distances = calculate_cosine_distances(centroid_vec, np_embedding_vectors)
                        
                        messages_with_distances = []
                        for i, pair_data in enumerate(message_pair_embeddings_with_ids):
                            messages_with_distances.append({
                                "user_msg_id": pair_data[0],
                                "assistant_msg_id": pair_data[1],
                                "distance": distances[i]
                            })
                        
                        # Sort by distance (normal focus: furthest first, inverted focus: closest first)
                        reverse_sort = False if score_mode == "inverted_focus" else True
                        sorted_messages_by_distance = sorted(messages_with_distances, key=lambda x: x["distance"], reverse=reverse_sort)
                        
                        agent.summarize_messages_focus_inplace(sorted_messages_by_distance)
                    else:
                        agent.summarize_messages_inplace()
                else:
                    agent.summarize_messages_inplace()
                    
            elif memory_mode == "hybrid":
                agent.summarize_messages_hybrid_inplace()
            else:  # FIFO mode
                agent.summarize_messages_inplace()
                
        except Exception as sum_e:
            return f"ERROR: MANUAL_SUMMARIZATION_FAILED: {sum_e}"

    # 4. Ask the final question
    final_question = test_case['question']
    final_question_json = json.dumps({"type": "user_message", "message": final_question})
    
    hypothesis = "ERROR: NO_RESPONSE_FROM_AGENT_STEP"
    try:
        response_messages, _, _, _, _ = agent.step(user_message=final_question_json, return_dicts=True)
        
        if response_messages and isinstance(response_messages, list):
            # Find the last assistant message in the response (in case there are function calls)
            assistant_message = None
            for msg in reversed(response_messages):
                if msg.get('role') == 'assistant' and msg.get('content'):
                    assistant_message = msg
                    break
            
            if assistant_message:
                hypothesis = assistant_message['content']
            else:
                hypothesis = "ERROR: NO_ASSISTANT_MESSAGE_FOUND"
        
    except Exception as e:
        hypothesis = f"ERROR: AGENT_STEP_FAILED: {e}"

    return hypothesis

def create_custom_message_pair_embeddings(message_sequence, embedding_config):
    """
    Create embeddings for user-assistant message pairs.
    """
    from memgpt.embeddings import create_embedding
    
    pair_embeddings = []
    if not message_sequence or len(message_sequence) < 2:
        return pair_embeddings

    for i in range(len(message_sequence) - 1):
        msg1 = message_sequence[i]
        msg2 = message_sequence[i+1]

        if msg1.role == "user" and msg2.role == "assistant":
            try:
                text1 = msg1.text if msg1.text is not None else ""
                text2 = msg2.text if msg2.text is not None else ""
                
                if not text1.strip() or not text2.strip():
                    continue
                
                if any(keyword in text1.lower() for keyword in ['login', 'bootup', 'system']):
                    continue
                
                if msg1.id is None or msg2.id is None:
                    continue

                combined_text = text1.strip() + " " + text2.strip()

                if len(combined_text.strip()) < 20:
                    continue

                try:
                    embedding_vector = create_embedding(
                        text=combined_text,
                        embedding_config=embedding_config,
                    )
                    pair_embeddings.append((msg1.id, msg2.id, embedding_vector))
                        
                except Exception as e:
                    continue
                    
            except Exception as e:
                continue

    return pair_embeddings

def write_results_safely(results_batch, output_path):
    """
    Thread-safe function to write results to output file.
    """
    with output_lock:
        with open(output_path, 'a', encoding='utf-8') as outfile:
            for result_data in results_batch:
                if result_data["success"]:
                    outfile.write(json.dumps(result_data["result"]) + '\n')
                    outfile.flush()

def main():
    """Main execution function with parallel case processing for temporal reasoning."""
    print("===== MemGPT LongMemEval TEMPORAL REASONING Case-Parallel Benchmark Script =====")
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run MemGPT LongMemEval benchmark with parallel case processing (TEMPORAL REASONING ONLY)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limited cases)")
    parser.add_argument("--mode", choices=["focus", "fifo", "hybrid"], default="focus", help="Memory mode")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for hybrid mode (0.0-1.0)")
    parser.add_argument("--cluster", action="store_true", help="Enable clustering-based summarization")
    parser.add_argument("--prompt-type", choices=["memgpt_default", "xml", "xml_temporal_reasoning"], default="xml_temporal_reasoning", help="Prompt type")
    parser.add_argument("--centroid-method", choices=["centroid", "medoid"], default="centroid", help="Centroid calculation method")
    parser.add_argument("--score-mode", choices=["inverted_focus"], help="Score mode modifier")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of test cases to run in parallel")
    parser.add_argument("--max-workers", type=int, help="Maximum number of worker processes (default: batch-size)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 <= args.beta <= 1.0):
        print(f"Error: Beta must be between 0.0 and 1.0, got {args.beta}")
        return
    
    if args.max_workers is None:
        args.max_workers = args.batch_size
    
    print(f"Configuration (TEMPORAL REASONING):")
    print(f"  Memory mode: {args.mode.upper()}")
    print(f"  Beta parameter: {args.beta}")
    print(f"  Clustering: {'ENABLED' if args.cluster else 'DISABLED'}")
    print(f"  Prompt type: {args.prompt_type}")
    print(f"  Centroid method: {args.centroid_method.upper()}")
    print(f"  Score mode: {args.score_mode.upper() if args.score_mode else 'NONE'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Test mode: {'ON' if args.test else 'OFF'}")
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"memgpt_hypotheses_temporal_reasoning_{args.prompt_type}_{args.mode}"
    if args.mode == "hybrid":
        output_filename += f"_beta{args.beta}"
    if args.cluster:
        output_filename += "_cluster"
    output_filename += f"_{args.centroid_method}"
    if args.score_mode:
        output_filename += f"_{args.score_mode}"
    if args.test:
        output_filename += "_test"
    output_filename += f"_batch{args.batch_size}_{timestamp}"
    output_path = os.path.join(MODULE_BASE_PATH, f"{output_filename}.jsonl")
    
    print(f"Output file: {output_path}")
    
    # Load config and data
    config = MemGPTConfig.load()
    config_dict = config.__dict__  # Convert to dict for multiprocessing
    
    test_cases = load_and_filter_data()
    
    # Filter for test mode  
    if args.test:
        test_cases = test_cases[0:10]  # Use fewer cases for test mode since temporal reasoning might be limited
        print(f"Test mode: Limited to {len(test_cases)} temporal reasoning cases")
    
    # --- RESUME LOGIC ---
    completed_question_ids = set()
    if os.path.exists(output_path) and not args.test:
        print(f"Previous run detected. Checking existing results...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'question_id' in data:
                            completed_question_ids.add(data['question_id'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading existing results: {e}. Starting fresh.")
            completed_question_ids = set()

    remaining_test_cases = [case for case in test_cases if case['question_id'] not in completed_question_ids]
    
    if completed_question_ids and not args.test:
        print(f"Found {len(completed_question_ids)} completed instances. Resuming with {len(remaining_test_cases)} remaining temporal reasoning cases.")
    else:
        print(f"Starting {'test' if args.test else 'benchmark'} run on {len(remaining_test_cases)} temporal reasoning cases.")
    
    if not remaining_test_cases:
        print("No remaining temporal reasoning test cases to process!")
        return
    
    # --- PARALLEL PROCESSING ---
    print(f"\nStarting parallel processing with {args.max_workers} workers, batch size {args.batch_size}...")
    
    # Prepare arguments for workers
    worker_args = []
    for i, test_case in enumerate(remaining_test_cases):
        worker_id = i % args.max_workers
        args_tuple = (
            test_case, config_dict, args.mode, args.beta, args.cluster,
            args.prompt_type, args.centroid_method, args.score_mode, worker_id
        )
        worker_args.append(args_tuple)
    
    # Process in batches
    total_processed = 0
    total_errors = 0
    
    # Create output file if it doesn't exist
    if args.test or not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Process in batches
            for batch_start in tqdm(range(0, len(worker_args), args.batch_size), desc="Processing temporal reasoning batches"):
                batch_end = min(batch_start + args.batch_size, len(worker_args))
                batch_args = worker_args[batch_start:batch_end]
                
                # Submit batch jobs
                future_to_args = {executor.submit(run_single_test_case, args): args for args in batch_args}
                
                # Collect results for this batch
                batch_results = []
                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result["success"]:
                            total_processed += 1
                        else:
                            total_errors += 1
                            print(f"Error in temporal reasoning case {result['question_id']}: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        total_errors += 1
                        print(f"Future execution error: {e}")
                
                # Write results for this batch
                write_results_safely(batch_results, output_path)
                
                # Progress update
                batch_num = batch_start // args.batch_size + 1
                total_batches = (len(worker_args) + args.batch_size - 1) // args.batch_size
                print(f"Completed temporal reasoning batch {batch_num}/{total_batches}. "
                      f"Total processed: {total_processed}, Errors: {total_errors}")
                      
    except KeyboardInterrupt:
        print("\n\nTemporal reasoning benchmark interrupted by user (Ctrl+C). Progress has been saved.")
        print("To resume, simply run the script again.")
    
    print(f"\n===== TEMPORAL REASONING {'TEST' if args.test else 'BENCHMARK'} RUN COMPLETE =====")
    print(f"Successfully processed: {total_processed} temporal reasoning cases")
    print(f"Errors encountered: {total_errors} cases")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    main() 