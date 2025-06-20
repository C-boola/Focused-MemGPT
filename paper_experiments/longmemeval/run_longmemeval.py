import json
import os
import uuid
from tqdm import tqdm
import time

# --- Debug Logging Utility ---
# Set DEBUG to True to see detailed step-by-step logs.
# Set to False for a clean, production-style run.
DEBUG = True

def log_debug(message):
    """A simple, toggleable logging function."""
    if DEBUG:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}")

# --- MemGPT and Benchmark Imports ---
from memgpt import create_client
from memgpt.constants import MESSAGE_SUMMARY_WARNING_FRAC
from memgpt.utils import count_tokens

# --- Constants ---
MODULE_BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_s.json")
ORACLE_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_oracle.json")
OUTPUT_PATH = os.path.join(MODULE_BASE_PATH, "memgpt_hypotheses.jsonl")

LONGMEM_PERSONA = "You are a helpful assistant with a perfect memory. You will answer questions based on the entire history of your conversation with a user."

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
    
    # Convert oracle list to dictionary for efficient lookup
    oracle_data = {item['question_id']: item for item in oracle_list}
    log_debug(f"Oracle data contains {len(oracle_data)} question entries.")
    
    filtered_data = [
        item for item in full_data
        if oracle_data.get(item['question_id'], {}).get('question_type') in FOCUSED_QUESTION_TYPES
    ]
    
    log_debug(f"Original data size: {len(full_data)} cases.")
    log_debug(f"Filtered to {len(filtered_data)} cases for types: {FOCUSED_QUESTION_TYPES}")
    if not filtered_data:
        print("CRITICAL WARNING: No test cases were loaded after filtering. Please check data files and question types.")
    return filtered_data


def run_test_instance(client, test_case: dict) -> str:
    """
    Manually constructs an agent's memory state, then asks the final question.
    """
    q_id = test_case['question_id']
    agent_name = f"longmemeval_agent_{q_id}"
    log_debug(f"--- Starting Test Instance: {q_id} ---")
    
    # 1. Agent Lifecycle: Cleanup and Create
    try:
        if client.get_agent(agent_name=agent_name):
             client.delete_agent(agent_name=agent_name)
             log_debug(f"Cleaned up pre-existing agent: {agent_name}")
    except Exception:
        pass 

    agent_state = client.create_agent(name=agent_name, persona=LONGMEM_PERSONA)
    log_debug(f"Created fresh agent '{agent_name}' with ID {agent_state.id}")

    try:
        agent_object = client.server._get_or_load_agent(user_id=agent_state.user_id, agent_id=agent_state.id)
        llm_context_window = agent_object.agent_state.llm_config.context_window
        log_debug(f"Successfully retrieved agent object. Context window size: {llm_context_window} tokens.")
    except Exception as e:
        log_debug(f"FATAL ERROR in instance {q_id}: Could not retrieve agent object. Error: {e}")
        return f"ERROR: FAILED TO GET AGENT OBJECT FOR {q_id}"
    
    # 2. Manual History Construction & Proactive Memory Management
    full_chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    log_debug(f"Beginning surgical memory injection of {len(full_chat_history)} turns.")
    for i, turn_dict in enumerate(full_chat_history):
        log_debug(f"  Injecting Turn {i+1}/{len(full_chat_history)} - Role: {turn_dict['role']}")
        agent_object.append_to_messages([turn_dict])
        
        # Verify the append by checking the last message
        last_msg = agent_object.messages[-1]
        if last_msg['content'] != turn_dict['content']:
             log_debug(f"  !! VERIFICATION FAILED: Appended message content does not match source.")
        
        # Check for memory pressure
        current_tokens = count_tokens(" ".join([msg['content'] for msg in agent_object.messages if msg.get('content')]))
        if current_tokens > MESSAGE_SUMMARY_WARNING_FRAC * llm_context_window:
            log_debug(f"  CONTEXT PRESSURE DETECTED: {current_tokens}/{llm_context_window} tokens. Triggering manual summarization.")
            agent_object.summarize_messages_inplace()
            new_token_count = count_tokens(" ".join([msg['content'] for msg in agent_object.messages if msg.get('content')]))
            log_debug(f"  Summarization complete. Token count reduced to {new_token_count}.")

    # 3. Trigger Final Reasoning
    final_question = test_case['question']
    log_debug(f"Memory priming complete. Asking final question: '{final_question}'")
    final_response_list = client.user_message(agent_id=agent_state.id, message=final_question)

    # 4. Extract Hypothesis
    hypothesis = "ERROR: NO ASSISTANT MESSAGE FOUND IN RESPONSE"
    if final_response_list:
        for msg in reversed(final_response_list):
            if msg.get("assistant_message"):
                hypothesis = msg["assistant_message"]
                log_debug(f"Extracted hypothesis: '{hypothesis[:100]}...'")
                break
    
    # 5. Cleanup
    client.delete_agent(agent_id=agent_state.id)
    log_debug(f"Test instance {q_id} complete. Cleaned up agent {agent_name}.")
    log_debug(f"--- Finished Test Instance: {q_id} ---\n")
    
    return hypothesis


def main():
    """Main execution function to run the benchmark."""
    print("===== MemGPT LongMemEval Benchmark Script =====")
    log_debug("Initializing MemGPT client...")
    client = create_client()
    
    test_cases = load_and_filter_data()
    
    print(f"\nStarting benchmark run on {len(test_cases)} filtered test cases.")
    log_debug(f"Results will be written to: {OUTPUT_PATH}")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
        for case in tqdm(test_cases, desc="Overall Progress"):
            hypothesis = run_test_instance(client, case)
            
            result = {
                "question_id": case['question_id'],
                "hypothesis": hypothesis,
                "ground_truth": case.get('answer', 'N/A')
            }
            outfile.write(json.dumps(result) + '\n')
            
    print(f"\n===== BENCHMARK RUN COMPLETE =====")
    print(f"All hypotheses have been generated and saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main() 