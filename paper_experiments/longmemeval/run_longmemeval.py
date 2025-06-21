import json
import os
import uuid
from tqdm import tqdm
import time
import sys
import traceback

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
from memgpt.constants import DEFAULT_PRESET, DEFAULT_PERSONA, DEFAULT_HUMAN
from memgpt.utils import get_persona_text, get_human_text
from memgpt.prompts import gpt_system
from memgpt.presets.presets import generate_functions_json

# A dummy interface to suppress the agent's internal terminal output during the run
class SilentInterface(AgentInterface):
    def user_message(self, msg, msg_obj=None) -> None: pass
    def internal_monologue(self, msg, msg_obj=None) -> None: pass
    def assistant_message(self, msg, msg_obj=None) -> None: pass
    def function_message(self, msg, msg_obj=None) -> None: pass

# --- Constants ---
MODULE_BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_s.json")
ORACLE_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_oracle.json")
OUTPUT_PATH = os.path.join(MODULE_BASE_PATH, "memgpt_hypotheses.jsonl")

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

def run_test_instance(base_config: MemGPTConfig, test_case: dict, memory_mode: str = "focus") -> str:
    """
    Directly instantiates and controls a MemGPT Agent to run a test case.
    """
    q_id = test_case['question_id']
    agent_name = f"longmemeval_agent_{q_id}"
    log_debug(f"--- Starting Test Instance: {q_id} (Memory Mode: {memory_mode}) ---")
    
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
        },
    )

    try:
        agent = Agent(interface=SilentInterface(), agent_state=agent_state)
        log_debug(f"Successfully created agent '{agent.agent_state.name}' in-memory.")
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

def main():
    """Main execution function to run the benchmark with resume and interrupt support."""
    print("===== MemGPT LongMemEval Benchmark Script =====")
    # Load the base config once to pass to each agent instance
    config = MemGPTConfig.load()
    
    test_cases = load_and_filter_data()
    
    # --- RESUME LOGIC ---
    completed_question_ids = set()
    if os.path.exists(OUTPUT_PATH):
        log_debug(f"Previous run detected. Checking existing results in: {OUTPUT_PATH}")
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        if 'question_id' in data:
                            completed_question_ids.add(data['question_id'])
                    except json.JSONDecodeError:
                        log_debug(f"Warning: Corrupted line {line_num+1} found in {OUTPUT_PATH}. Skipping.")
        except Exception as e:
            log_debug(f"Error reading existing results file {OUTPUT_PATH}: {e}. Starting fresh.")
            completed_question_ids = set()

    remaining_test_cases = [case for case in test_cases if case['question_id'] not in completed_question_ids]
    
    if completed_question_ids:
        print(f"\nFound {len(completed_question_ids)} completed instances. Resuming with {len(remaining_test_cases)} remaining test cases.")
    else:
        print(f"\nStarting a fresh benchmark run on {len(test_cases)} filtered test cases.")
    
    # --- GRACEFUL INTERRUPT & APPEND LOGIC ---
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as outfile:
        try:
            for case in tqdm(remaining_test_cases, desc="Overall Progress"):
                hypothesis = run_test_instance(config, case)
                
                result = {
                    "question_id": case['question_id'],
                    "hypothesis": hypothesis,
                    "ground_truth": case.get('answer', 'N/A') 
                }
                outfile.write(json.dumps(result) + '\n')
                outfile.flush()
                
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user (Ctrl+C). Progress has been saved.")
            print("To resume, simply run the script again.")
            
    print(f"\n===== BENCHMARK RUN COMPLETE =====")
    print(f"All hypotheses have been generated and saved to:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    main()