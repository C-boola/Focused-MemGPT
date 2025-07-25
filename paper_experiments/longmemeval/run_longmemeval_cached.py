import json
import os
import uuid
from tqdm import tqdm
import time
import sys
import traceback
from datetime import datetime

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

# Import embedding utilities
from embedding_utils import EmbeddingCacheManager, check_embedding_cache_coverage

# A dummy interface to suppress the agent's internal terminal output during the run
class SilentInterface(AgentInterface):
    def user_message(self, msg, msg_obj=None) -> None: 
        # Show user messages if they contain memory warnings or heartbeats
        if msg and ("memory" in msg.lower() or "heartbeat" in msg.lower() or "summary" in msg.lower()):
            print(f"[MEMORY] User: {msg}")
    
    def internal_monologue(self, msg, msg_obj=None) -> None: 
        # Show internal thoughts related to memory operations
        if msg and any(keyword in msg.lower() for keyword in ["memory", "summary", "context", "overflow", "embedding"]):
            print(f"[MEMORY] Internal: {msg}")
    
    def assistant_message(self, msg, msg_obj=None) -> None: 
        # Show assistant messages related to memory operations
        if msg and any(keyword in msg.lower() for keyword in ["memory", "summary", "context", "overflow"]):
            print(f"[MEMORY] Assistant: {msg}")
    
    def function_message(self, msg, msg_obj=None) -> None: 
        # Show all function calls related to memory
        if msg and any(keyword in msg.lower() for keyword in ["memory", "summary", "archival", "core_memory", "recall"]):
            print(f"[MEMORY] Function: {msg}")
        # Also show any running/success messages for memory functions
        elif msg and ("running" in msg.lower() or "success" in msg.lower()) and any(keyword in msg.lower() for keyword in ["memory", "archival", "core"]):
            print(f"[MEMORY] Function: {msg}")

# --- Constants ---
memory_mode = "hybrid"

MODULE_BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_s.json")
ORACLE_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_oracle.json")

FOCUSED_QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session"
}

def load_and_filter_data() -> list[dict]:
    """Loads and filters LongMemEval data for the focused question types."""
    log_debug("Executing load_and_filter_data...")
    log_debug(f"Loading main data from: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    log_debug(f"Loading oracle data from: {ORACLE_PATH}")
    with open(ORACLE_PATH, 'r', encoding='utf-8') as f:
        oracle_list = json.load(f)
    
    oracle_data = {item['question_id']: item for item in oracle_list}
    log_debug(f"Oracle data contains {len(oracle_data)} question entries.")
    
    filtered_data = [
        item for item in full_data
        if oracle_data.get(item['question_id'], {}).get('question_type') in FOCUSED_QUESTION_TYPES
    ]
    
    log_debug(f"Original data size: {len(full_data)} cases.")
    log_debug(f"Filtered to {len(filtered_data)} cases for types: {', '.join(FOCUSED_QUESTION_TYPES)}")
    if not filtered_data:
        print("CRITICAL WARNING: No test cases were loaded after filtering. Please check data files and question types.")
    return filtered_data

def create_optimized_agent_with_cached_embeddings(base_config: MemGPTConfig, test_case: dict, memory_mode: str, beta: float, cluster_summaries: bool, prompt_type: str, centroid_method: str, score_mode: str, cache_manager: EmbeddingCacheManager):
    """
    Create an agent with optimized embedding handling using cached embeddings when possible.
    """
    q_id = test_case['question_id']
    agent_name = f"longmemeval_agent_{q_id}"
    
    # Create agent state
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
        agent = Agent(interface=SilentInterface(), agent_state=agent_state, mem_mode=memory_mode, beta=beta, cluster_summaries=cluster_summaries, centroid_method=centroid_method, score_mode=score_mode)
        log_debug(f"Successfully created agent '{agent.agent_state.name}' with memory mode: {memory_mode}")
        
        # Check if embeddings are available for this question
        has_cached_embeddings = cache_manager.has_embeddings_for_question(q_id, base_config.default_embedding_config)
        
        if has_cached_embeddings:
            log_debug(f"Found cached embeddings for question {q_id} - will use for optimization")
            # Pre-load embeddings for potential use
            cached_embeddings = cache_manager.get_embeddings_for_question(q_id, base_config.default_embedding_config)
            # Store in agent for potential use during summarization
            if hasattr(agent, '_cached_embeddings'):
                agent._cached_embeddings = cached_embeddings
            else:
                # If the agent doesn't have this attribute, we'll add it dynamically
                setattr(agent, '_cached_embeddings', cached_embeddings)
            log_debug(f"Pre-loaded {len(cached_embeddings) if cached_embeddings else 0} cached embeddings")
        else:
            log_debug(f"No cached embeddings found for question {q_id} - will generate on-demand")
            setattr(agent, '_cached_embeddings', None)
        
        return agent
        
    except Exception as e:
        log_debug(f"FATAL ERROR in instance {q_id}: Could not instantiate Agent object. Error: {e}")
        traceback.print_exc()
        return None

def run_test_instance_with_cache(base_config: MemGPTConfig, test_case: dict, memory_mode: str = "focus", beta: float = 0.5, cluster_summaries: bool = False, prompt_type: str = "memgpt_default", centroid_method: str = "centroid", score_mode: str = None, cache_manager: EmbeddingCacheManager = None) -> str:
    """
    Run a test instance with optimized embedding caching.
    """
    q_id = test_case['question_id']
    log_debug(f"--- Starting Test Instance: {q_id} (Memory Mode: {memory_mode}, Beta: {beta}, Clustering: {'ON' if cluster_summaries else 'OFF'}, Prompt Type: {prompt_type}, Centroid Method: {centroid_method}, Score Mode: {score_mode or 'None'}) ---")
    
    # Create optimized agent
    agent = create_optimized_agent_with_cached_embeddings(
        base_config, test_case, memory_mode, beta, cluster_summaries, 
        prompt_type, centroid_method, score_mode, cache_manager
    )
    
    if agent is None:
        return f"ERROR: AGENT_INSTANTIATION_FAILED FOR {q_id}"
    
    # Inject conversation history
    full_chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    log_debug(f"Beginning surgical memory injection of {len(full_chat_history)} turns.")
    
    agent.append_to_messages(full_chat_history)
    log_debug(f"Injection complete. Agent now has {len(agent.messages)} messages in its history.")

    # Check context overflow and force summarization if needed
    log_debug("Checking if injected history exceeds context limits...")
    
    current_tokens = sum(count_tokens(str(msg)) for msg in agent.messages)
    context_window = agent.agent_state.llm_config.context_window
    overflow_threshold = MESSAGE_SUMMARY_WARNING_FRAC * context_window
    
    log_debug(f"Current tokens: {current_tokens}, Context window: {context_window}, Overflow threshold: {overflow_threshold}")
    
    if current_tokens > overflow_threshold:
        log_debug(f"CONTEXT OVERFLOW DETECTED! Current tokens ({current_tokens}) > threshold ({overflow_threshold})")
        log_debug(f"Forcing {memory_mode} mode summarization...")
        
        try:
            if memory_mode == "focus":
                mode_name = "INVERTED FOCUS MODE" if score_mode == "inverted_focus" else "FOCUS MODE"
                print("=" * 70)
                print(f"  ||  MANUAL {mode_name} SUMMARIZATION TRIGGERED (CACHED OPTIMIZED)  ||")
                print("=" * 70)
                
                # Try to use cached embeddings first
                if hasattr(agent, '_cached_embeddings') and agent._cached_embeddings:
                    log_debug(f"Using pre-cached embeddings for focus mode ({len(agent._cached_embeddings)} pairs)")
                    message_pair_embeddings_with_ids = agent._cached_embeddings
                else:
                    log_debug("No cached embeddings available, generating on-the-fly")
                    message_pair_embeddings_with_ids = agent._create_robust_message_pair_embeddings(
                        message_sequence=agent._messages, 
                        embedding_config=agent.agent_state.embedding_config
                    )
                
                log_debug(f"Using {len(message_pair_embeddings_with_ids)} message pair embeddings for focus mode.")
                
                if message_pair_embeddings_with_ids:
                    from memgpt.embeddings import calculate_centroid, calculate_medoid, calculate_cosine_distances
                    import numpy as np
                    
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
                        
                        reverse_sort = False if score_mode == "inverted_focus" else True
                        sorted_messages_by_distance = sorted(messages_with_distances, key=lambda x: x["distance"], reverse=reverse_sort)
                        log_debug(f"Sorted {len(sorted_messages_by_distance)} message pairs by distance.")
                        
                        agent.summarize_messages_focus_inplace(sorted_messages_by_distance)
                        log_debug(f"{mode_name} summarization completed successfully.")
                    else:
                        log_debug(f"{centroid_method.capitalize()} calculation failed, falling back to FIFO.")
                        agent.summarize_messages_inplace()
                else:
                    log_debug("No embeddings available, falling back to FIFO.")
                    agent.summarize_messages_inplace()
                    
            elif memory_mode == "hybrid":
                print("=" * 70)
                print("  ||  MANUAL HYBRID MODE SUMMARIZATION TRIGGERED (CACHED OPTIMIZED)  ||")
                print("=" * 70)
                
                # Pre-load cached embeddings if available for hybrid mode
                if hasattr(agent, '_cached_embeddings') and agent._cached_embeddings and beta > 0.0:
                    log_debug(f"Pre-loaded cached embeddings for hybrid mode available ({len(agent._cached_embeddings)} pairs)")
                    # Note: The hybrid summarization will check for and use these cached embeddings
                elif beta == 0.0:
                    log_debug("Beta=0 detected, embeddings not needed for pure FIFO hybrid mode")
                else:
                    log_debug("No cached embeddings available for hybrid mode, will generate on-demand")
                
                agent.summarize_messages_hybrid_inplace()
                log_debug("Hybrid mode summarization completed successfully.")
                
            elif memory_mode == "pure_cluster":
                print("=" * 70)
                print("  ||  MANUAL PURE CLUSTER MODE SUMMARIZATION TRIGGERED (CACHED OPTIMIZED)  ||")
                print("=" * 70)
                agent.summarize_messages_pure_cluster_inplace()
                log_debug("Pure cluster mode summarization completed successfully.")
                
            elif memory_mode == "density":
                print("=" * 70)
                print("  ||  MANUAL DENSITY MODE SUMMARIZATION TRIGGERED (CACHED OPTIMIZED)  ||")
                print("=" * 70)
                agent.summarize_messages_density_inplace()
                log_debug("Density mode summarization completed successfully.")
                
            else:  # FIFO mode
                print("=" * 70)
                print("  ||  MANUAL FIFO MODE SUMMARIZATION TRIGGERED  ||")
                print("=" * 70)
                agent.summarize_messages_inplace()
                log_debug("FIFO mode summarization completed successfully.")
                
            # Check tokens after summarization
            new_tokens = sum(count_tokens(str(msg)) for msg in agent.messages)
            log_debug(f"After summarization: {len(agent.messages)} messages, {new_tokens} tokens (reduced by {current_tokens - new_tokens})")
            
        except Exception as sum_e:
            log_debug(f"ERROR during manual summarization: {sum_e}")
            traceback.print_exc()
            return f"ERROR: MANUAL_SUMMARIZATION_FAILED: {sum_e}"
    else:
        log_debug("No context overflow detected. Proceeding without summarization.")

    # Trigger reasoning with the final question
    final_question = test_case['question']
    log_debug(f"Memory priming complete. Sending final question to trigger agent.step(): '{final_question}'")
    
    final_question_json = json.dumps({"type": "user_message", "message": final_question})
    
    hypothesis = "ERROR: NO_RESPONSE_FROM_AGENT_STEP"
    try:
        response_messages, _, _, _, _ = agent.step(user_message=final_question_json, return_dicts=True)
        
        # Extract hypothesis
        log_debug(f"Raw response from agent.step(): {response_messages}")
        if response_messages and isinstance(response_messages, list):
            assistant_message = None
            for msg in reversed(response_messages):
                if msg.get('role') == 'assistant' and msg.get('content'):
                    assistant_message = msg
                    break
            
            if assistant_message:
                hypothesis = assistant_message['content']
                log_debug(f"Extracted hypothesis: '{hypothesis[:100]}...'")
            else:
                hypothesis = "ERROR: NO_ASSISTANT_MESSAGE_FOUND"
        
    except Exception as e:
        log_debug(f"ERROR in instance {q_id}: agent.step() failed. Exception: {e}")
        traceback.print_exc()
        hypothesis = f"ERROR: AGENT_STEP_FAILED: {e}"

    log_debug(f"--- Finished Test Instance: {q_id} ---\n")
    return hypothesis

def main():
    """Main execution function with embedding cache optimization."""
    print("===== MemGPT LongMemEval Benchmark Script (Cache-Optimized) =====")
    
    # Parse arguments
    args = sys.argv[1:]
    test_mode = "--test" in args
    
    memory_mode = "focus"  # Default mode
    if "--mode" in args:
        try:
            mode_index = args.index("--mode") + 1
            if mode_index < len(args):
                specified_mode = args[mode_index]
                if specified_mode in ["focus", "fifo", "hybrid", "pure_cluster", "density"]:
                    memory_mode = specified_mode
                else:
                    print(f"Warning: Invalid memory mode '{specified_mode}'. Defaulting to 'focus'.")
            else:
                print("Warning: --mode flag used without a value. Defaulting to 'focus'.")
        except (ValueError, IndexError):
            print("Error parsing --mode flag. Defaulting to 'focus'.")

    beta = 0.5  # Default beta
    if "--beta" in args:
        try:
            beta_index = args.index("--beta") + 1
            if beta_index < len(args):
                specified_beta = float(args[beta_index])
                if 0.0 <= specified_beta <= 1.0:
                    beta = specified_beta
                else:
                    print(f"Warning: Beta must be between 0.0 and 1.0, got {specified_beta}. Defaulting to 0.5.")
            else:
                print("Warning: --beta flag used without a value. Defaulting to 0.5.")
        except (ValueError, IndexError):
            print("Error parsing --beta flag. Defaulting to 0.5.")

    cluster_summaries = False
    if "--cluster" in args:
        try:
            cluster_index = args.index("--cluster") + 1
            if cluster_index < len(args):
                specified_cluster = args[cluster_index].lower()
                if specified_cluster == "true":
                    cluster_summaries = True
                elif specified_cluster == "false":
                    cluster_summaries = False
                else:
                    print(f"Warning: Invalid cluster value '{args[cluster_index]}'. Defaulting to False.")
            else:
                print("Warning: --cluster flag used without a value. Defaulting to False.")
        except (ValueError, IndexError):
            print("Error parsing --cluster flag. Defaulting to False.")

    prompt_type = "xml"
    if "--prompt-type" in args:
        try:
            prompt_type_index = args.index("--prompt-type") + 1
            if prompt_type_index < len(args):
                specified_prompt_type = args[prompt_type_index]
                if specified_prompt_type in ["memgpt_default", "xml", "xml_temporal_reasoning"]:
                    prompt_type = specified_prompt_type
                else:
                    print(f"Warning: Invalid prompt type '{specified_prompt_type}'. Defaulting to 'xml'.")
            else:
                print("Warning: --prompt-type flag used without a value. Defaulting to 'xml'.")
        except (ValueError, IndexError):
            print("Error parsing --prompt-type flag. Defaulting to 'xml'.")

    centroid_method = "centroid"
    if "--centroid-method" in args:
        try:
            centroid_method_index = args.index("--centroid-method") + 1
            if centroid_method_index < len(args):
                specified_centroid_method = args[centroid_method_index]
                if specified_centroid_method in ["centroid", "medoid"]:
                    centroid_method = specified_centroid_method
                else:
                    print(f"Warning: Invalid centroid method '{specified_centroid_method}'. Defaulting to 'centroid'.")
            else:
                print("Warning: --centroid-method flag used without a value. Defaulting to 'centroid'.")
        except (ValueError, IndexError):
            print("Error parsing --centroid-method flag. Defaulting to 'centroid'.")

    score_mode = None
    if "--score-mode" in args:
        try:
            score_mode_index = args.index("--score-mode") + 1
            if score_mode_index < len(args):
                specified_score_mode = args[score_mode_index]
                if specified_score_mode in ["inverted_focus"]:
                    score_mode = specified_score_mode
                else:
                    print(f"Warning: Invalid score mode '{specified_score_mode}'. Defaulting to None.")
            else:
                print("Warning: --score-mode flag used without a value. Defaulting to None.")
        except (ValueError, IndexError):
            print("Error parsing --score-mode flag. Defaulting to None.")

    trunc_frac = 0.75
    if "--trunc-frac" in args:
        try:
            trunc_frac_index = args.index("--trunc-frac") + 1
            if trunc_frac_index < len(args):
                specified_trunc_frac = float(args[trunc_frac_index])
                if 0.0 < specified_trunc_frac <= 1.0:
                    trunc_frac = specified_trunc_frac
                else:
                    print(f"Warning: Truncation fraction must be between 0.0 and 1.0, got {specified_trunc_frac}. Defaulting to 0.75.")
            else:
                print("Warning: --trunc-frac flag used without a value. Defaulting to 0.75.")
        except (ValueError, IndexError):
            print("Error parsing --trunc-frac flag. Defaulting to 0.75.")

    # Set the global truncation fraction constant
    import memgpt.constants
    memgpt.constants.MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC = trunc_frac
    print(f"Set MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC to {trunc_frac}")

    # Initialize embedding cache manager
    print("\n--- Embedding Cache Analysis ---")
    cache_manager = EmbeddingCacheManager()
    
    if cache_manager.is_cache_available():
        cache_info = cache_manager.get_cache_info()
        print("✅ Embedding cache is available!")
        for model_hash, model_info in cache_info["models"].items():
            print(f"   Model {model_info['model']}: {model_info['num_questions']} questions cached")
    else:
        print("❌ Embedding cache not available. Embeddings will be generated on-demand.")
        print("   Run 'python precompute_embeddings.py' to build the cache.")
    
    # Load configuration and test data
    config = MemGPTConfig.load()
    test_cases = load_and_filter_data()
    
    # Check cache coverage
    if cache_manager.is_cache_available():
        all_question_ids = [case['question_id'] for case in test_cases]
        coverage = check_embedding_cache_coverage(all_question_ids, config.default_embedding_config)
        print(f"\n📊 Cache Coverage: {coverage['cached_questions']}/{coverage['total_questions']} ({coverage['coverage_percentage']:.1f}%)")
        
        # Show which embedding-dependent modes will benefit
        embedding_dependent_modes = ["focus", "hybrid", "pure_cluster", "density"]
        will_benefit = memory_mode in embedding_dependent_modes and (memory_mode != "hybrid" or beta > 0.0)
        
        if will_benefit:
            if coverage['coverage_percentage'] == 100:
                print("🚀 Selected mode will use cached embeddings for maximum performance!")
            elif coverage['coverage_percentage'] > 0:
                print(f"⚡ Selected mode will use cached embeddings for {coverage['cached_questions']} cases")
                print(f"   {len(coverage['missing_questions'])} cases will generate embeddings on-demand")
            else:
                print("⚠️  Selected mode requires embeddings but none are cached")
        else:
            print("ℹ️  Selected mode doesn't require embeddings (FIFO or hybrid with beta=0)")
    
    if test_mode:
        print("\nRUNNING IN TEST MODE - Will process only first 3 cases with verbose output")
        test_cases = test_cases[0:3]
    
    print(f"Using memory mode: {memory_mode.upper()}")
    if memory_mode == "hybrid":
        print(f"Using beta parameter: {beta}")
    print(f"Clustering-based summarization: {'ENABLED' if cluster_summaries else 'DISABLED'}")
    print(f"Prompt type: {prompt_type}")
    print(f"Centroid method: {centroid_method.upper()}")
    print(f"Score mode: {score_mode.upper() if score_mode else 'NONE'}")
    print(f"Truncation fraction: {trunc_frac}")
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"memgpt_hypotheses_cached_{prompt_type}_{memory_mode}"
    if memory_mode == "hybrid":
        output_filename += f"_beta{beta}"
    if cluster_summaries:
        output_filename += "_cluster"
    output_filename += f"_{centroid_method}"
    if score_mode:
        output_filename += f"_{score_mode}"
    output_filename += f"_trunc{trunc_frac}"
    if test_mode:
        output_filename += "_test"
    output_filename += f"_{timestamp}"
    output_path = os.path.join(MODULE_BASE_PATH, f"{output_filename}.jsonl")
    
    # Resume logic
    completed_question_ids = set()
    if os.path.exists(output_path) and not test_mode:
        log_debug(f"Previous run detected. Checking existing results in: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        if 'question_id' in data:
                            completed_question_ids.add(data['question_id'])
                    except json.JSONDecodeError:
                        log_debug(f"Warning: Corrupted line {line_num+1} found in {output_path}. Skipping.")
        except Exception as e:
            log_debug(f"Error reading existing results file {output_path}: {e}. Starting fresh.")
            completed_question_ids = set()

    remaining_test_cases = [case for case in test_cases if case['question_id'] not in completed_question_ids]
    
    if completed_question_ids and not test_mode:
        print(f"\nFound {len(completed_question_ids)} completed instances. Resuming with {len(remaining_test_cases)} remaining test cases.")
    else:
        print(f"\nStarting {'test' if test_mode else 'benchmark'} run on {len(remaining_test_cases)} {'test' if test_mode else 'filtered'} cases.")
    
    # Run benchmark with cache optimization
    with open(output_path, 'w' if test_mode else 'a', encoding='utf-8') as outfile:
        try:
            for case in tqdm(remaining_test_cases, desc="Overall Progress"):
                if test_mode:
                    print(f"\n{'='*50}")
                    print(f"TEST CASE: {case['question_id']}")
                    print(f"QUESTION: {case['question']}")
                    print(f"SESSIONS: {len(case['haystack_sessions'])}")
                    total_turns = sum(len(session) for session in case['haystack_sessions'])
                    print(f"TOTAL TURNS: {total_turns}")
                    
                    # Show cache status for this question
                    has_cache = cache_manager.has_embeddings_for_question(case['question_id'], config.default_embedding_config)
                    print(f"CACHED EMBEDDINGS: {'✅ Available' if has_cache else '❌ Not cached'}")
                    print(f"{'='*50}")
                
                hypothesis = run_test_instance_with_cache(
                    config, case, memory_mode=memory_mode, beta=beta, 
                    cluster_summaries=cluster_summaries, prompt_type=prompt_type, 
                    centroid_method=centroid_method, score_mode=score_mode,
                    cache_manager=cache_manager
                )
                
                result = {
                    "question_id": case['question_id'],
                    "hypothesis": hypothesis,
                    "ground_truth": case.get('answer', 'N/A') 
                }
                outfile.write(json.dumps(result) + '\n')
                outfile.flush()
                
                if test_mode:
                    print(f"RESULT: {hypothesis}")
                    print(f"GROUND TRUTH: {result['ground_truth']}")
                
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user (Ctrl+C). Progress has been saved.")
            print("To resume, simply run the script again.")
            
    print(f"\n===== {'TEST' if test_mode else 'BENCHMARK'} RUN COMPLETE =====")
    print(f"All hypotheses have been generated and saved to:\n{output_path}")

if __name__ == "__main__":
    main() 