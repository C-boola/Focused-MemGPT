import json
import os
import uuid
from tqdm import tqdm
import time
import sys
import traceback
from datetime import datetime

# --- Debug Logging Utility ---
# Set DEBUG to True to see detailed step-by-step logs.
# Set to False for a clean, production-style run.
DEBUG = True

def log_debug(message):
    """A simple, toggleable logging function."""
    if DEBUG:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}")

# --- MemGPT and Benchmark Imports ---
# Direct imports for manual agent creation and control
from memgpt.agent import Agent
from memgpt.interface import AgentInterface
from memgpt.data_types import AgentState
from memgpt.presets.presets import available_presets
from memgpt.config import MemGPTConfig
from memgpt.constants import DEFAULT_PRESET, DEFAULT_PERSONA, DEFAULT_HUMAN, MESSAGE_SUMMARY_WARNING_FRAC
from memgpt.utils import get_persona_text, get_human_text, count_tokens
from memgpt.prompts import gpt_system
from memgpt.presets.presets import generate_functions_json

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
# OUTPUT_PATH will be set dynamically in main() based on mode and beta

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

def run_test_instance(base_config: MemGPTConfig, test_case: dict, memory_mode: str = "focus", beta: float = 0.5, cluster_summaries: bool = False, prompt_type: str = "memgpt_default", centroid_method: str = "centroid", score_mode: str = None) -> str:
    """
    Directly instantiates and controls a MemGPT Agent to run a test case.
    """
    q_id = test_case['question_id']
    agent_name = f"longmemeval_agent_{q_id}"
    log_debug(f"--- Starting Test Instance: {q_id} (Memory Mode: {memory_mode}, Beta: {beta}, Clustering: {'ON' if cluster_summaries else 'OFF'}, Prompt Type: {prompt_type}, Centroid Method: {centroid_method}, Score Mode: {score_mode or 'None'}) ---")
    
    # 1. Direct Agent Creation (in-memory)
    dummy_user_id = uuid.uuid4()
    # Use a standard preset to get default system prompts and functions
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
            "mem_mode": memory_mode,  # Explicitly set memory mode
            "beta": beta,  # Explicitly set beta parameter
            "cluster_summaries": cluster_summaries,  # Explicitly set clustering parameter
            "prompt_type": prompt_type,  # Explicitly set prompt type
            "centroid_method": centroid_method,  # Explicitly set centroid method
            "score_mode": score_mode,  # Explicitly set score mode
        },
    )

    try:
        agent = Agent(interface=SilentInterface(), agent_state=agent_state, mem_mode=memory_mode, beta=beta, cluster_summaries=cluster_summaries, centroid_method=centroid_method, score_mode=score_mode)
        log_debug(f"Successfully created agent '{agent.agent_state.name}' with memory mode: {memory_mode}, beta: {beta}, clustering: {'ON' if cluster_summaries else 'OFF'}, prompt_type: {prompt_type}, centroid_method: {centroid_method}, score_mode: {score_mode or 'None'}")
    except Exception as e:
        log_debug(f"FATAL ERROR in instance {q_id}: Could not instantiate Agent object. Error: {e}")
        traceback.print_exc()
        return f"ERROR: AGENT_INSTANTIATION_FAILED FOR {q_id}"
    
    # 2. Manual History Construction
    full_chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    log_debug(f"Beginning surgical memory injection of {len(full_chat_history)} turns.")
    
    # Use the agent's internal, direct append method
    agent.append_to_messages(full_chat_history)
    log_debug(f"Injection complete. Agent now has {len(agent.messages)} messages in its history.")

    # 2.5. EXPLICIT CONTEXT OVERFLOW CHECK AND FORCE SUMMARIZATION
    log_debug("Checking if injected history exceeds context limits...")
    
    # Calculate current token count
    current_tokens = sum(count_tokens(str(msg)) for msg in agent.messages)
    context_window = agent.agent_state.llm_config.context_window
    overflow_threshold = MESSAGE_SUMMARY_WARNING_FRAC * context_window
    
    log_debug(f"Current tokens: {current_tokens}, Context window: {context_window}, Overflow threshold: {overflow_threshold}")
    
    if current_tokens > overflow_threshold:
        log_debug(f"CONTEXT OVERFLOW DETECTED! Current tokens ({current_tokens}) > threshold ({overflow_threshold})")
        log_debug(f"Forcing {memory_mode} mode summarization...")
        
        try:
            if memory_mode == "focus":
                # Force focus summarization
                print("=" * 70)
                mode_name = "INVERTED FOCUS MODE" if score_mode == "inverted_focus" else "FOCUS MODE"
                print(f"  ||  MANUAL {mode_name} SUMMARIZATION TRIGGERED  ||")
                print("=" * 70)
                
                from memgpt.embeddings import calculate_centroid, calculate_medoid, calculate_cosine_distances
                import numpy as np
                
                # Generate embeddings for focus mode using our custom function
                message_pair_embeddings_with_ids = create_custom_message_pair_embeddings(
                    message_sequence=agent._messages, 
                    embedding_config=agent.agent_state.embedding_config
                )
                log_debug(f"Generated {len(message_pair_embeddings_with_ids)} message pair embeddings.")
                
                # DEBUG: Let's see what the message structure looks like
                if len(message_pair_embeddings_with_ids) == 0:
                    log_debug("DEBUG: No embeddings generated. Analyzing message structure...")
                    log_debug(f"Total messages in agent: {len(agent._messages)}")
                    
                    # Check first few messages to understand structure
                    for i, msg in enumerate(agent._messages[:10]):
                        log_debug(f"Message {i}: role='{msg.role}', text_snippet='{str(msg.text)[:100] if msg.text else 'None'}...'")
                    
                    # Check for user-assistant pairs manually
                    user_assistant_pairs = 0
                    for i in range(len(agent._messages) - 1):
                        msg1 = agent._messages[i]
                        msg2 = agent._messages[i+1]
                        if msg1.role == "user" and msg2.role == "assistant":
                            user_assistant_pairs += 1
                            if user_assistant_pairs <= 3:  # Show first 3 pairs
                                log_debug(f"Found pair {user_assistant_pairs}: User='{str(msg1.text)[:50]}...', Assistant='{str(msg2.text)[:50]}...'")
                    
                    log_debug(f"Found {user_assistant_pairs} user-assistant pairs manually")
                
                if message_pair_embeddings_with_ids:
                    # Calculate centroid/medoid and distances
                    actual_embedding_vectors = [pair[2] for pair in message_pair_embeddings_with_ids]
                    if centroid_method == "medoid":
                        centroid_vec = calculate_medoid(actual_embedding_vectors)
                    else:
                        centroid_vec = calculate_centroid(actual_embedding_vectors)
                    
                    if centroid_vec is not None:
                        np_embedding_vectors = [np.array(vec) for vec in actual_embedding_vectors]
                        distances = calculate_cosine_distances(centroid_vec, np_embedding_vectors)
                        
                        # Create sorted messages by distance
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
                        sort_desc = "closest first" if score_mode == "inverted_focus" else "furthest first"
                        log_debug(f"Sorted {len(sorted_messages_by_distance)} message pairs by distance ({sort_desc}).")
                        
                        # Trigger focus summarization
                        agent.summarize_messages_focus_inplace(sorted_messages_by_distance)
                        log_debug(f"{mode_name} summarization completed successfully.")
                    else:
                        log_debug(f"{centroid_method.capitalize()} calculation failed, falling back to FIFO.")
                        agent.summarize_messages_inplace()
                else:
                    log_debug("No embeddings generated, falling back to FIFO.")
                    agent.summarize_messages_inplace()
                    
            elif memory_mode == "hybrid":
                print("=" * 70)
                print("  ||  MANUAL HYBRID MODE SUMMARIZATION TRIGGERED  ||")
                print("=" * 70)
                agent.summarize_messages_hybrid_inplace()
                log_debug("Hybrid mode summarization completed successfully.")
                
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

    # 3. Trigger Reasoning with the Final Question
    final_question = test_case['question']
    log_debug(f"Memory priming complete. Sending final question to trigger agent.step(): '{final_question}'")
    
    # Format the question as JSON to work around MemGPT's agent.step() expecting JSON format
    final_question_json = json.dumps({"type": "user_message", "message": final_question})
    
    hypothesis = "ERROR: NO_RESPONSE_FROM_AGENT_STEP"
    try:
        # This single call to step() will run the full, authentic internal MemGPT logic:
        # - It will construct a prompt with the full history we injected.
        # - It will likely get a ContextOverflowError from the LLM API.
        # - It will catch this error and trigger the appropriate memory summarization
        #   (FIFO or Focus mode), which includes pair embedding.
        # - It will then retry the LLM call with the compressed context.
        response_messages, _, _, _, _ = agent.step(user_message=final_question_json, return_dicts=True)
        
        # 4. Extract Hypothesis
        log_debug(f"Raw response from agent.step(): {response_messages}")
        if response_messages and isinstance(response_messages, list):
            last_message = response_messages[-1]
            if last_message['role'] == 'assistant':
                hypothesis = last_message['content']
                log_debug(f"Extracted hypothesis: '{hypothesis[:100]}...'")
            else:
                hypothesis = "ERROR: FINAL_MESSAGE_NOT_FROM_ASSISTANT"
        
    except Exception as e:
        log_debug(f"ERROR in instance {q_id}: agent.step() failed. This could be a persistent context overflow or another API error. Exception: {e}")
        traceback.print_exc()
        hypothesis = f"ERROR: AGENT_STEP_FAILED: {e}"

    log_debug(f"--- Finished Test Instance: {q_id} ---\n")
    # Agent object is now out of scope and will be garbage collected. No manual deletion needed.
    return hypothesis

def create_custom_message_pair_embeddings(message_sequence, embedding_config):
    """
    Custom embedding function for plain text messages (not JSON).
    Creates vector embeddings for message pairs (user message and subsequent assistant message).
    """
    from memgpt.embeddings import create_embedding
    import uuid
    
    pair_embeddings = []
    if not message_sequence or len(message_sequence) < 2:
        return pair_embeddings

    log_debug(f"Creating custom embeddings for {len(message_sequence)} messages...")
    
    for i in range(len(message_sequence) - 1):
        msg1 = message_sequence[i]
        msg2 = message_sequence[i+1]

        # Look for user-assistant pairs with plain text
        if msg1.role == "user" and msg2.role == "assistant":
            try:
                # Get text content directly (no JSON parsing needed)
                text1 = msg1.text if msg1.text is not None else ""
                text2 = msg2.text if msg2.text is not None else ""
                
                # Skip empty messages or system-like messages
                if not text1.strip() or not text2.strip():
                    continue
                
                # Skip login messages and other system messages
                if any(keyword in text1.lower() for keyword in ['login', 'bootup', 'system']):
                    continue
                
                # Ensure IDs are not None
                if msg1.id is None or msg2.id is None:
                    log_debug(f"Warning: Skipping message pair due to missing ID(s). User msg ID: {msg1.id}, Assistant msg ID: {msg2.id}")
                    continue

                # Combine the messages
                combined_text = text1.strip() + " " + text2.strip()

                # Skip if combined text is too short
                if len(combined_text.strip()) < 20:
                    continue

                try:
                    # Create embedding for the combined text
                    embedding_vector = create_embedding(
                        text=combined_text,
                        embedding_config=embedding_config,
                    )
                    pair_embeddings.append((msg1.id, msg2.id, embedding_vector))
                        
                except Exception as e:
                    log_debug(f"Error creating embedding for pair (user_msg_id={msg1.id}, assistant_msg_id={msg2.id}): {e}")
                    continue
                    
            except Exception as e:
                log_debug(f"Error processing message pair at index {i}: {e}")
                continue

    log_debug(f"Successfully created {len(pair_embeddings)} message pair embeddings")
    return pair_embeddings

def main():
    """Main execution function to run the benchmark with resume and interrupt support."""
    print("===== MemGPT LongMemEval Benchmark Script =====")
    
    # --- Argument Parsing ---
    args = sys.argv[1:]
    test_mode = "--test" in args
    
    memory_mode = "focus"  # Default mode
    if "--mode" in args:
        try:
            mode_index = args.index("--mode") + 1
            if mode_index < len(args):
                specified_mode = args[mode_index]
                if specified_mode in ["focus", "fifo", "hybrid"]:
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
            print("Error parsing --beta flag (must be a number between 0.0 and 1.0). Defaulting to 0.5.")

    cluster_summaries = "--cluster" in args  # Default is False (clustering OFF)

    prompt_type = "memgpt_default"  # Default prompt type
    if "--prompt-type" in args:
        try:
            prompt_type_index = args.index("--prompt-type") + 1
            if prompt_type_index < len(args):
                specified_prompt_type = args[prompt_type_index]
                if specified_prompt_type in ["memgpt_default", "xml"]:
                    prompt_type = specified_prompt_type
                else:
                    print(f"Warning: Invalid prompt type '{specified_prompt_type}'. Defaulting to 'memgpt_default'.")
            else:
                print("Warning: --prompt-type flag used without a value. Defaulting to 'memgpt_default'.")
        except (ValueError, IndexError):
            print("Error parsing --prompt-type flag. Defaulting to 'memgpt_default'.")

    centroid_method = "centroid"  # Default centroid method
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

    score_mode = None  # Default score mode
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

    if test_mode:
        print("RUNNING IN TEST MODE - Will process only first 3 cases with verbose output")
    print(f"Using memory mode: {memory_mode.upper()}")
    if memory_mode == "hybrid":
        print(f"Using beta parameter: {beta}")
    else:
        print(f"Beta parameter: {beta} (only used in hybrid mode)")
    print(f"Clustering-based summarization: {'ENABLED' if cluster_summaries else 'DISABLED'}")
    print(f"Prompt type: {prompt_type}")
    print(f"Centroid method: {centroid_method.upper()}")
    print(f"Score mode: {score_mode.upper() if score_mode else 'NONE'}")
    
    # Create output path based on mode, beta, clustering, centroid method, score mode, and prompt type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"memgpt_hypotheses_{prompt_type}_{memory_mode}"
    if memory_mode == "hybrid":
        output_filename += f"_beta{beta}"
    if cluster_summaries:
        output_filename += "_cluster"
    output_filename += f"_{centroid_method}"
    if score_mode:
        output_filename += f"_{score_mode}"
    if test_mode:
        output_filename += "_test"
    output_filename += f"_{timestamp}"
    output_path = os.path.join(MODULE_BASE_PATH, f"{output_filename}.jsonl")
    
    # Load the base config once to pass to each agent instance
    config = MemGPTConfig.load()
    
    test_cases = load_and_filter_data()
    
    # In test mode, only run first 3 cases
    if test_mode:
        test_cases = test_cases[8:15]
        print(f"Test mode: Limited to {len(test_cases)} cases")
    
    # --- RESUME LOGIC ---
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
    
    # Use different output file for test mode
    final_output_path = output_path #+ ".test" if test_mode else output_path
    
    # --- GRACEFUL INTERRUPT & APPEND LOGIC ---
    with open(final_output_path, 'w' if test_mode else 'a', encoding='utf-8') as outfile:
        try:
            for case in tqdm(remaining_test_cases, desc="Overall Progress"):
                if test_mode:
                    print(f"\n{'='*50}")
                    print(f"TEST CASE: {case['question_id']}")
                    print(f"QUESTION: {case['question']}")
                    print(f"SESSIONS: {len(case['haystack_sessions'])}")
                    total_turns = sum(len(session) for session in case['haystack_sessions'])
                    print(f"TOTAL TURNS: {total_turns}")
                    print(f"{'='*50}")
                
                hypothesis = run_test_instance(config, case, memory_mode=memory_mode, beta=beta, cluster_summaries=cluster_summaries, prompt_type=prompt_type, centroid_method=centroid_method, score_mode=score_mode)
                
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
    print(f"All hypotheses have been generated and saved to:\n{final_output_path}")

if __name__ == "__main__":
    main()