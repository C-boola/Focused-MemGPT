import json
import os
import uuid
from tqdm import tqdm
import time
import sys
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
from memgpt.llm_api_tools import create as create_llm_completion
import openai
from memgpt.config import MemGPTConfig
from memgpt.credentials import MemGPTCredentials
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

class CallbackInterface(SilentInterface):
    """An interface that captures and stores function calls made by the agent."""
    def __init__(self):
        self.function_calls = []
        super().__init__()

    def function_message(self, msg, msg_obj=None) -> None:
        # Call the parent class's method to still get the memory logs
        super().function_message(msg, msg_obj)
        
        # Specifically capture the 'Running' messages for search functions
        if msg and "running" in msg.lower() and ("archival_memory_search" in msg.lower() or "conversation_search" in msg.lower()):
            self.function_calls.append(msg)
    
    def clear_calls(self):
        self.function_calls = []

# --- Constants ---
memory_mode = "hybrid"

MODULE_BASE_PATH = os.path.dirname(__file__)
# --- Constants for Integrated Benchmark ---
REFLECT_PERSONA_PATH = os.path.join(os.path.dirname(__file__), "data", "persona_reflect_and_archive.txt")
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

def run_test_instance(
    base_config: MemGPTConfig,
    test_case: dict,
    memory_mode: str = "focus",
    beta: float = 0.5,
    cluster_summaries: bool = False,
    prompt_type: str = "memgpt_default",
    centroid_method: str = "centroid",
    score_mode: str = None,
) -> str:
    """
    Runs a single longmemeval test using the 'Direct Tool Invocation' strategy.
    """
    q_id = test_case['question_id']
    agent_name = f"longmemeval_agent_{q_id}"
    log_debug(f"--- Starting Integrated Test Instance: {q_id} ---")

    # 1. Create agent with default persona; answering logic is now in the core system prompt
    answer_persona_text = get_persona_text(DEFAULT_PERSONA)

    preset_config = available_presets[DEFAULT_PRESET]
    agent_state = AgentState(
        name=agent_name, user_id=uuid.uuid4(), persona=answer_persona_text,
        human=get_human_text(DEFAULT_HUMAN), preset=DEFAULT_PRESET,
        llm_config=base_config.default_llm_config, embedding_config=base_config.default_embedding_config,
        state={
            "persona": answer_persona_text, "human": get_human_text(DEFAULT_HUMAN),
            "system": gpt_system.get_system_text(preset_config["system_prompt"]),
            "functions": generate_functions_json(preset_config["functions"]), "messages": None,
            "mem_mode": memory_mode, "beta": beta, "cluster_summaries": cluster_summaries,
            "prompt_type": prompt_type, "centroid_method": centroid_method, "score_mode": score_mode,
        },
    )

    callback_interface = CallbackInterface()
    try:
        agent = Agent(interface=callback_interface, agent_state=agent_state)
        log_debug(f"Agent initialized with default persona and system prompt-based search protocol.")
    except Exception as e:
        log_debug(f"FATAL ERROR: Could not instantiate Agent object. Error: {e}")
        return f"ERROR: AGENT_INSTANTIATION_FAILED"

    # 2. Start the "Direct Tool Invocation" phase to populate memory
    log_debug("\n===== STARTING DIRECT MEMORY POPULATION PHASE =====")
    full_chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    conversation_text_block = "\n".join([f"{turn['role']}: {turn['content']}" for turn in full_chat_history])
    chunk_size = 8000  # Use larger chunks to reduce LLM calls
    text_chunks = [conversation_text_block[i:i+chunk_size] for i in range(0, len(conversation_text_block), chunk_size)]
    log_debug(f"Divided conversation into {len(text_chunks)} chunks for fact extraction.")

    # Define a simple function schema for fact extraction
    extract_facts_schema = {
        "name": "extract_facts",
        "description": "Extract core identity facts and archivable knowledge from conversation text",
        "parameters": {
            "type": "object",
            "properties": {
                "core_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Permanent, foundational facts about the user's identity, personality, or stated long-term goals"
                },
                "archivable_facts": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Specific, detailed pieces of information, events, or topics discussed that are important to remember"
                }
            },
            "required": ["core_facts", "archivable_facts"]
        }
    }

    # Define a function to process a single chunk in parallel
    def process_chunk(chunk_data):
        """Process a single chunk and return extracted facts."""
        chunk_index, chunk = chunk_data
        try:
            extraction_prompt = f"""Analyze the following conversation text and extract two types of facts:
1. **core_facts**: Permanent, foundational facts about the user's identity, personality, or stated long-term goals.
2. **archivable_facts**: Specific, detailed pieces of information, events, or topics discussed that are important to remember.

Return a JSON object containing these two lists.

Conversation Text:
```
{chunk}
```"""
            
            messages = [{"role": "user", "content": extraction_prompt}]

            # Initialize the OpenAI client (thread-safe)
            credentials = MemGPTCredentials.load()
            client = openai.OpenAI(api_key=credentials.openai_key)

            # Direct, modern API call
            response = client.chat.completions.create(
                model=agent.agent_state.llm_config.model,
                messages=messages,
                tools=[{"type": "function", "function": extract_facts_schema}],
                tool_choice={"type": "function", "function": {"name": "extract_facts"}},
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                return {
                    "chunk_index": chunk_index,
                    "core_facts": args.get("core_facts", []),
                    "archivable_facts": args.get("archivable_facts", []),
                    "success": True
                }
            else:
                return {
                    "chunk_index": chunk_index,
                    "core_facts": [],
                    "archivable_facts": [],
                    "success": False,
                    "error": "No tool calls in response"
                }
            
        except Exception as e:
            return {
                "chunk_index": chunk_index,
                "core_facts": [],
                "archivable_facts": [],
                "success": False,
                "error": str(e)
            }

    # Process chunks in parallel with ThreadPoolExecutor
    log_debug(f"Processing {len(text_chunks)} chunks in parallel for faster fact extraction...")
    max_workers = min(8, len(text_chunks))  # Limit concurrent requests to avoid rate limits
    chunk_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk processing tasks
        chunk_data = [(i, chunk) for i, chunk in enumerate(text_chunks)]
        future_to_chunk = {executor.submit(process_chunk, data): data for data in chunk_data}
        
        # Collect results as they complete with progress bar
        for future in tqdm(as_completed(future_to_chunk), total=len(text_chunks), desc="Extracting facts from history"):
            result = future.result()
            chunk_results.append(result)
            
            if not result["success"]:
                log_debug(f"Warning: Failed to extract facts from chunk {result['chunk_index']}: {result.get('error', 'Unknown error')}")

    # Sort results by chunk index to maintain order
    chunk_results.sort(key=lambda x: x["chunk_index"])
    
    # Collect all facts for parallel memory saving
    log_debug("Preparing facts for parallel memory saving...")
    all_core_facts = []
    all_archivable_facts = []
    
    for result in chunk_results:
        if result["success"]:
            all_core_facts.extend(result["core_facts"])
            all_archivable_facts.extend(result["archivable_facts"])
    
    # Thread-safe memory saving with locks
    core_memory_lock = threading.Lock()
    archival_memory_lock = threading.Lock()
    core_memory_full = threading.Event()  # Signal when core memory is full
    memory_stats = {"core_saved": 0, "archival_saved": 0, "core_fallback": 0, "errors": 0}
    stats_lock = threading.Lock()
    
    def save_core_fact(fact):
        """Thread-safe core memory saving with fallback logic."""
        try:
            with core_memory_lock:
                # Check if core memory was already flagged as full
                if core_memory_full.is_set():
                    log_debug(f"Core Memory already full, routing directly to Archival: {fact}")
                    save_to_archival_fallback(fact)
                    return
                
                try:
                    log_debug(f"Attempting to save to Core Memory: {fact}")
                    agent.memory.edit_append('human', fact)
                    log_debug(f"Successfully saved to Core Memory.")
                    with stats_lock:
                        memory_stats["core_saved"] += 1
                except ValueError as e:
                    if "exceeds character limit" in str(e).lower():
                        log_debug(f"Core Memory full detected. Setting flag and routing to Archival: {fact}")
                        core_memory_full.set()  # Signal that core memory is full
                        save_to_archival_fallback(fact)
                    else:
                        log_debug(f"Warning: Core memory append failed with non-size error: {e}")
                        with stats_lock:
                            memory_stats["errors"] += 1
        except Exception as e:
            log_debug(f"Warning: Unexpected error during core memory save: {e}")
            with stats_lock:
                memory_stats["errors"] += 1
    
    def save_to_archival_fallback(fact):
        """Save fact to archival memory as fallback from core memory."""
        try:
            with archival_memory_lock:
                agent.persistence_manager.archival_memory.insert(fact)
                log_debug(f"Successfully saved to Archival Memory as fallback.")
                with stats_lock:
                    memory_stats["core_fallback"] += 1
        except Exception as e:
            log_debug(f"Warning: Archival memory fallback failed: {e}")
            with stats_lock:
                memory_stats["errors"] += 1
    
    def save_archival_fact(fact):
        """Thread-safe archival memory saving."""
        try:
            with archival_memory_lock:
                log_debug(f"Saving to Archival Memory: {fact}")
                agent.persistence_manager.archival_memory.insert(fact)
                with stats_lock:
                    memory_stats["archival_saved"] += 1
        except Exception as e:
            log_debug(f"Warning: Archival memory insert failed: {e}")
            with stats_lock:
                memory_stats["errors"] += 1
    
    # Phase 1: Save core facts in parallel (with intelligent fallback)
    if all_core_facts:
        log_debug(f"Saving {len(all_core_facts)} core facts in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:  # Smaller pool for memory ops
            core_futures = [executor.submit(save_core_fact, fact) for fact in all_core_facts]
            for future in tqdm(as_completed(core_futures), total=len(all_core_facts), desc="Saving core facts"):
                future.result()  # Wait for completion and handle any exceptions
    
    # Phase 2: Save archival facts in parallel
    if all_archivable_facts:
        log_debug(f"Saving {len(all_archivable_facts)} archival facts in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:  # Smaller pool for memory ops
            archival_futures = [executor.submit(save_archival_fact, fact) for fact in all_archivable_facts]
            for future in tqdm(as_completed(archival_futures), total=len(all_archivable_facts), desc="Saving archival facts"):
                future.result()  # Wait for completion and handle any exceptions
    
    # Report final statistics
    log_debug(f"Parallel memory saving complete:")
    log_debug(f"  Core facts saved: {memory_stats['core_saved']}")
    log_debug(f"  Archival facts saved: {memory_stats['archival_saved']}")
    log_debug(f"  Core->Archival fallbacks: {memory_stats['core_fallback']}")
    log_debug(f"  Errors encountered: {memory_stats['errors']}")
    
    total_saved = memory_stats['core_saved'] + memory_stats['archival_saved'] + memory_stats['core_fallback']
    log_debug(f"Total facts successfully saved: {total_saved}/{len(all_core_facts) + len(all_archivable_facts)}")

    log_debug("===== DIRECT MEMORY POPULATION COMPLETE =====\n")

    # 3. Prepare for the final question: Load the full history to trigger natural context overflow
    log_debug("Loading full conversation history to trigger context management...")
    agent.append_to_messages(full_chat_history)
    log_debug(f"Agent context loaded with {len(agent.messages)} messages.")

    # 4. Ask the final question
    final_question = test_case['question']
    log_debug(f"Asking final question: '{final_question}'")
    
    # CRITICAL: Clear any previous calls from the reflection phase before the final question
    callback_interface.clear_calls()
    
    hypothesis = "ERROR: NO_RESPONSE_FROM_AGENT_STEP"
    try:
        response_messages, _, _, _, _ = agent.step(user_message=json.dumps({'role': 'user', 'content': final_question}), return_dicts=True)
        
        # After the step, check and log the function calls that were made
        log_debug("\n===== FINAL STEP ANALYSIS =====")
        if callback_interface.function_calls:
            print("[ANALYSIS] Agent made the following memory search calls during the final step:")
            for call in callback_interface.function_calls:
                print(f"  -> {call}")
        else:
            print("[ANALYSIS] Agent did NOT make any memory search calls. Answer was likely from summarized context.")
        log_debug("=============================\n")

        log_debug(f"Raw response from agent.step(): {response_messages}")
        if response_messages and isinstance(response_messages, list):
            assistant_message = next((msg['content'] for msg in reversed(response_messages) if msg.get('role') == 'assistant' and msg.get('content')), None)
            if assistant_message:
                hypothesis = assistant_message
                log_debug(f"Extracted hypothesis: '{hypothesis[:100]}...'")
            else:
                hypothesis = "ERROR: NO_ASSISTANT_MESSAGE_FOUND"
    except Exception as e:
        log_debug(f"ERROR in instance {q_id}: agent.step() failed. Exception: {e}")
        traceback.print_exc()
        hypothesis = f"ERROR: AGENT_STEP_FAILED: {e}"

    log_debug(f"--- Finished Integrated Test Instance: {q_id} ---\n")
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
    """Main execution function to run the integrated benchmark with resume and interrupt support."""
    print("===== MemGPT LongMemEval Direct Tool Invocation Benchmark Script =====")
    
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

    prompt_type = "xml"  # Default prompt type
    if "--prompt-type" in args:
        try:
            prompt_type_index = args.index("--prompt-type") + 1
            if prompt_type_index < len(args):
                specified_prompt_type = args[prompt_type_index]
                if specified_prompt_type in ["memgpt_default", "xml", "xml_temporal_reasoning"]:
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
        test_cases = test_cases[0:25]
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
            
    print(f"\n===== {'DIRECT TOOL INVOCATION TEST' if test_mode else 'DIRECT TOOL INVOCATION BENCHMARK'} RUN COMPLETE =====")
    print(f"All hypotheses have been generated and saved to:\n{final_output_path}")

if __name__ == "__main__":
    main()