#!/usr/bin/env python3
"""
Diagnostic script to analyze why beta sweeps aren't producing different accuracies.
Run this on a subset of your data to understand what's happening.
"""

import json
import os
import uuid
import numpy as np
from memgpt.agent import Agent
from memgpt.interface import AgentInterface
from memgpt.data_types import AgentState
from memgpt.config import MemGPTConfig
from memgpt.constants import DEFAULT_PRESET, DEFAULT_PERSONA, DEFAULT_HUMAN
from memgpt.utils import get_persona_text, get_human_text, count_tokens
from memgpt.prompts import gpt_system
from memgpt.presets.presets import generate_functions_json, available_presets
from memgpt.embeddings import calculate_centroid, calculate_cosine_distances

class SilentInterface(AgentInterface):
    def user_message(self, msg, msg_obj=None) -> None: pass
    def internal_monologue(self, msg, msg_obj=None) -> None: pass
    def assistant_message(self, msg, msg_obj=None) -> None: pass
    def function_message(self, msg, msg_obj=None) -> None: pass

def create_test_agent(base_config, memory_mode="hybrid", beta=0.5):
    """Create a test agent for diagnostics"""
    dummy_user_id = uuid.uuid4()
    preset_config = available_presets[DEFAULT_PRESET]
    preset_system_prompt = preset_config["system_prompt"]
    preset_function_set_names = preset_config["functions"]
    functions_schema = generate_functions_json(preset_function_set_names)
    
    agent_state = AgentState(
        name=f"diagnostic_agent_{uuid.uuid4().hex[:8]}",
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
            "cluster_summaries": False,
            "prompt_type": "memgpt_default",
        },
    )
    
    return Agent(interface=SilentInterface(), agent_state=agent_state, mem_mode=memory_mode, beta=beta)

def analyze_message_pair_selection(agent, chat_history):
    """Analyze how different beta values affect message pair selection"""
    
    # Inject history
    agent.append_to_messages(chat_history)
    print(f"Injected {len(chat_history)} messages, agent has {len(agent.messages)} total messages")
    
    # Generate embeddings
    message_pair_embeddings_with_ids = agent._create_robust_message_pair_embeddings(
        message_sequence=agent._messages, 
        embedding_config=agent.agent_state.embedding_config
    )
    
    print(f"Generated {len(message_pair_embeddings_with_ids)} message pair embeddings")
    print(f"Filtering ratio: {len(message_pair_embeddings_with_ids)} embeddings from {len(agent._messages)} total messages")
    
    if not message_pair_embeddings_with_ids:
        print("❌ No embeddings generated - this is the problem!")
        return
    
    # Calculate focus scores
    actual_embedding_vectors = [pair[2] for pair in message_pair_embeddings_with_ids]
    centroid_vec = calculate_centroid(actual_embedding_vectors)
    
    if centroid_vec is None:
        print("❌ Centroid calculation failed!")
        return
    
    np_embedding_vectors = [np.array(vec) for vec in actual_embedding_vectors]
    distances = calculate_cosine_distances(centroid_vec, np_embedding_vectors)
    
    # Calculate FIFO scores (CURRENT BUGGY METHOD)
    total_pairs = len(message_pair_embeddings_with_ids)
    fifo_scores_buggy = []
    for i in range(total_pairs):
        fifo_score = (total_pairs - i) / total_pairs
        fifo_scores_buggy.append(fifo_score)
    
    # Calculate CORRECT FIFO scores based on actual message positions
    message_id_to_position = {msg.id: idx for idx, msg in enumerate(agent._messages)}
    fifo_scores_correct = []
    for user_id, asst_id, _ in message_pair_embeddings_with_ids:
        user_pos = message_id_to_position.get(user_id, 0)
        asst_pos = message_id_to_position.get(asst_id, 0)
        avg_pos = (user_pos + asst_pos) / 2
        # Normalize: older messages (lower position) get higher scores
        max_pos = len(agent._messages) - 1
        fifo_score_correct = (max_pos - avg_pos) / max_pos if max_pos > 0 else 0.0
        fifo_scores_correct.append(fifo_score_correct)
    
    # Normalize focus scores
    max_distance = max(distances) if distances else 1.0
    focus_scores = [d / max_distance for d in distances]
    
    print("\n" + "="*60)
    print("DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    print(f"\nFocus scores range: {min(focus_scores):.4f} to {max(focus_scores):.4f}")
    print(f"FIFO scores (buggy) range: {min(fifo_scores_buggy):.4f} to {max(fifo_scores_buggy):.4f}")
    print(f"FIFO scores (correct) range: {min(fifo_scores_correct):.4f} to {max(fifo_scores_correct):.4f}")
    
    # Check correlation between focus and FIFO
    focus_array = np.array(focus_scores)
    fifo_buggy_array = np.array(fifo_scores_buggy)
    fifo_correct_array = np.array(fifo_scores_correct)
    
    correlation_buggy = np.corrcoef(focus_array, fifo_buggy_array)[0, 1]
    correlation_correct = np.corrcoef(focus_array, fifo_correct_array)[0, 1]
    
    print(f"\nCorrelation between Focus and FIFO (buggy): {correlation_buggy:.4f}")
    print(f"Correlation between Focus and FIFO (correct): {correlation_correct:.4f}")
    
    if abs(correlation_buggy) > 0.7:
        print("⚠️  HIGH CORRELATION DETECTED - Focus and FIFO are selecting similar pairs!")
    
    # Test different beta values
    print(f"\nTesting hybrid scores with different beta values:")
    beta_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for beta in beta_values:
        hybrid_scores_buggy = []
        hybrid_scores_correct = []
        
        for i in range(len(focus_scores)):
            hybrid_buggy = focus_scores[i] * beta + fifo_scores_buggy[i] * (1.0 - beta)
            hybrid_correct = focus_scores[i] * beta + fifo_scores_correct[i] * (1.0 - beta)
            hybrid_scores_buggy.append(hybrid_buggy)
            hybrid_scores_correct.append(hybrid_correct)
        
        # Get top 10 pairs for removal
        top_10_indices_buggy = sorted(range(len(hybrid_scores_buggy)), 
                                     key=lambda i: hybrid_scores_buggy[i], reverse=True)[:10]
        top_10_indices_correct = sorted(range(len(hybrid_scores_correct)), 
                                       key=lambda i: hybrid_scores_correct[i], reverse=True)[:10]
        
        overlap = len(set(top_10_indices_buggy) & set(top_10_indices_correct))
        
        print(f"  Beta={beta:.1f}: Buggy vs Correct top-10 overlap: {overlap}/10 ({overlap/10*100:.0f}%)")
    
    # Show actual message positions for top pairs
    print(f"\nTop 5 pairs selected for removal (Focus only):")
    top_focus_indices = sorted(range(len(focus_scores)), key=lambda i: focus_scores[i], reverse=True)[:5]
    
    for i, idx in enumerate(top_focus_indices):
        user_id, asst_id, _ = message_pair_embeddings_with_ids[idx]
        user_pos = message_id_to_position.get(user_id, -1)
        asst_pos = message_id_to_position.get(asst_id, -1)
        print(f"  {i+1}. User msg pos: {user_pos}, Assistant msg pos: {asst_pos}, Focus score: {focus_scores[idx]:.4f}")
    
    print(f"\nTop 5 pairs selected for removal (FIFO only - correct):")
    top_fifo_indices = sorted(range(len(fifo_scores_correct)), key=lambda i: fifo_scores_correct[i], reverse=True)[:5]
    
    for i, idx in enumerate(top_fifo_indices):
        user_id, asst_id, _ = message_pair_embeddings_with_ids[idx]
        user_pos = message_id_to_position.get(user_id, -1)
        asst_pos = message_id_to_position.get(asst_id, -1)
        print(f"  {i+1}. User msg pos: {user_pos}, Assistant msg pos: {asst_pos}, FIFO score: {fifo_scores_correct[idx]:.4f}")

def main():
    """Run diagnostics on a sample case"""
    print("🔍 HYBRID MODE DIAGNOSTIC TOOL")
    print("="*60)
    
    # Load config and a sample test case
    config = MemGPTConfig.load()
    
    # Load a test case from your data
    data_path = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s.json")
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure you're running this from the longmemeval directory")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Use the first test case for analysis
    test_case = full_data[0]
    chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    
    print(f"Analyzing test case: {test_case['question_id']}")
    print(f"Question: {test_case['question']}")
    print(f"Chat history length: {len(chat_history)} turns")
    
    # Create agent and analyze
    agent = create_test_agent(config, memory_mode="hybrid", beta=0.5)
    analyze_message_pair_selection(agent, chat_history)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. Fix FIFO score calculation to use actual message positions")
    print("2. If correlation is high, consider different scoring approaches")
    print("3. Analyze why so many message pairs are filtered during embedding")
    print("4. Consider using actual conversation timestamps instead of positions")

if __name__ == "__main__":
    main() 