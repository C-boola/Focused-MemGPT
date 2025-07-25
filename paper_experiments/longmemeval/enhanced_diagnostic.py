#!/usr/bin/env python3
"""
Enhanced diagnostic to analyze why different beta values produce identical results.
Focus on token-based selection stopping criterion.
"""

import json
import os
import uuid
import numpy as np
from memgpt.agent import Agent
from memgpt.interface import AgentInterface
from memgpt.data_types import AgentState
from memgpt.config import MemGPTConfig
from memgpt.constants import DEFAULT_PRESET, DEFAULT_PERSONA, DEFAULT_HUMAN, MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC
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
        name=f"enhanced_diagnostic_agent_{uuid.uuid4().hex[:8]}",
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

def analyze_token_based_selection_issue(agent, chat_history):
    """Analyze the token-based selection stopping criterion"""
    
    # Inject history
    agent.append_to_messages(chat_history)
    print(f"Injected {len(chat_history)} messages, agent has {len(agent.messages)} total messages")
    
    # Calculate token requirements
    current_total_tokens = sum(count_tokens(str(msg)) for msg in agent.messages)
    context_window = agent.agent_state.llm_config.context_window
    target_token_count = int(context_window * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)
    tokens_to_free = current_total_tokens - target_token_count
    
    print(f"Current tokens: {current_total_tokens}")
    print(f"Target tokens: {target_token_count}")
    print(f"Tokens to free: {tokens_to_free}")
    
    # Generate embeddings
    message_pair_embeddings_with_ids = agent._create_robust_message_pair_embeddings(
        message_sequence=agent._messages, 
        embedding_config=agent.agent_state.embedding_config
    )
    
    if not message_pair_embeddings_with_ids:
        print("❌ No embeddings generated!")
        return
    
    # Calculate scores
    actual_embedding_vectors = [pair[2] for pair in message_pair_embeddings_with_ids]
    centroid_vec = calculate_centroid(actual_embedding_vectors)
    np_embedding_vectors = [np.array(vec) for vec in actual_embedding_vectors]
    distances = calculate_cosine_distances(centroid_vec, np_embedding_vectors)
    
    # Calculate focus scores
    max_distance = max(distances) if distances else 1.0
    focus_scores = {(pair[0], pair[1]): distances[i] / max_distance 
                   for i, pair in enumerate(message_pair_embeddings_with_ids)}
    
    # Calculate corrected FIFO scores
    message_id_to_position = {msg.id: idx for idx, msg in enumerate(agent._messages)}
    max_position = len(agent._messages) - 1
    fifo_scores = {}
    
    for user_msg_id, assistant_msg_id, _ in message_pair_embeddings_with_ids:
        user_pos = message_id_to_position.get(user_msg_id, 0)
        asst_pos = message_id_to_position.get(assistant_msg_id, 0)
        avg_position = (user_pos + asst_pos) / 2.0
        
        if max_position > 0:
            fifo_score = (max_position - avg_position) / max_position
        else:
            fifo_score = 0.0
        
        fifo_scores[(user_msg_id, assistant_msg_id)] = max(0.0, min(1.0, fifo_score))
    
    # Calculate token counts for each pair
    message_id_to_message_obj = {msg.id: msg for msg in agent._messages if msg.id is not None}
    pair_tokens = {}
    
    for user_msg_id, assistant_msg_id, _ in message_pair_embeddings_with_ids:
        user_msg_obj = message_id_to_message_obj.get(user_msg_id)
        asst_msg_obj = message_id_to_message_obj.get(assistant_msg_id)
        
        if user_msg_obj and asst_msg_obj:
            pair_token_count = (count_tokens(str(user_msg_obj.to_openai_dict())) + 
                               count_tokens(str(asst_msg_obj.to_openai_dict())))
            pair_tokens[(user_msg_id, assistant_msg_id)] = pair_token_count
    
    print(f"\nToken distribution in pairs:")
    token_values = list(pair_tokens.values())
    print(f"Min tokens per pair: {min(token_values)}")
    print(f"Max tokens per pair: {max(token_values)}")
    print(f"Mean tokens per pair: {np.mean(token_values):.1f}")
    print(f"Median tokens per pair: {np.median(token_values):.1f}")
    
    # Test different beta values with token-based stopping
    beta_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    print(f"\n" + "="*80)
    print("TOKEN-BASED SELECTION ANALYSIS")
    print("="*80)
    
    selection_results = {}
    
    for beta in beta_values:
        print(f"\n--- Beta = {beta} ---")
        
        # Calculate hybrid scores
        hybrid_scores = {}
        for pair_key in focus_scores.keys():
            focus_component = focus_scores[pair_key] * beta
            fifo_component = fifo_scores[pair_key] * (1.0 - beta)
            hybrid_scores[pair_key] = focus_component + fifo_component
        
        # Sort pairs by hybrid score (highest first)
        sorted_pairs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Simulate token-based selection
        selected_pairs = []
        tokens_freed_so_far = 0
        
        for (user_id, asst_id), score in sorted_pairs:
            pair_token_count = pair_tokens.get((user_id, asst_id), 0)
            
            selected_pairs.append({
                'pair': (user_id, asst_id),
                'score': score,
                'tokens': pair_token_count,
                'cumulative_tokens': tokens_freed_so_far + pair_token_count
            })
            
            tokens_freed_so_far += pair_token_count
            
            if tokens_freed_so_far >= tokens_to_free:
                print(f"Stopping after {len(selected_pairs)} pairs (freed {tokens_freed_so_far} tokens)")
                break
        
        selection_results[beta] = selected_pairs
        
        # Show first few selected pairs
        print(f"First 5 selected pairs:")
        for i, pair_info in enumerate(selected_pairs[:5]):
            user_pos = message_id_to_position.get(pair_info['pair'][0], -1)
            asst_pos = message_id_to_position.get(pair_info['pair'][1], -1)
            print(f"  {i+1}. Pos: ({user_pos}, {asst_pos}), Score: {pair_info['score']:.4f}, "
                  f"Tokens: {pair_info['tokens']}, Cumulative: {pair_info['cumulative_tokens']}")
    
    # Compare selections across beta values
    print(f"\n" + "="*80)
    print("BETA COMPARISON ANALYSIS")
    print("="*80)
    
    # Check if the same pairs are selected
    for i, beta1 in enumerate(beta_values):
        for beta2 in beta_values[i+1:]:
            pairs1 = set(p['pair'] for p in selection_results[beta1])
            pairs2 = set(p['pair'] for p in selection_results[beta2])
            
            overlap = len(pairs1 & pairs2)
            total_unique = len(pairs1 | pairs2)
            
            print(f"Beta {beta1} vs Beta {beta2}: {overlap}/{max(len(pairs1), len(pairs2))} pairs overlap "
                  f"({overlap/max(len(pairs1), len(pairs2))*100:.1f}%)")
    
    # Analyze if early high-scoring pairs dominate
    print(f"\n" + "="*80)
    print("EARLY SELECTION DOMINANCE ANALYSIS")
    print("="*80)
    
    # Check if top-scoring pairs have enough tokens to meet threshold
    for beta in [0.0, 1.0]:  # Pure FIFO vs Pure Focus
        pairs_data = selection_results[beta]
        
        print(f"\nBeta {beta} (Pure {'FIFO' if beta == 0.0 else 'Focus'}):")
        
        # Calculate what percentage of tokens come from first few pairs
        total_tokens_selected = sum(p['tokens'] for p in pairs_data)
        
        for n in [1, 3, 5, 10]:
            if len(pairs_data) >= n:
                first_n_tokens = sum(p['tokens'] for p in pairs_data[:n])
                percentage = (first_n_tokens / total_tokens_selected) * 100
                print(f"  First {n} pairs contain {first_n_tokens}/{total_tokens_selected} tokens ({percentage:.1f}%)")
    
    # Recommend solutions
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(set(len(selection_results[beta]) for beta in beta_values)) == 1:
        print("❌ All beta values select the same number of pairs")
        
        avg_pair_tokens = np.mean(token_values)
        pairs_needed = int(tokens_to_free / avg_pair_tokens) + 1
        
        print(f"💡 SOLUTION OPTIONS:")
        print(f"1. Use FIXED number of pairs instead of token threshold")
        print(f"   - Estimate: {pairs_needed} pairs needed based on average pair size")
        print(f"2. Use PERCENTAGE of pairs instead (e.g., remove oldest 25%)")
        print(f"3. Set MINIMUM pairs threshold even if tokens are met early")
        print(f"4. Use stratified sampling across score ranges")
    else:
        print("✅ Different beta values select different numbers of pairs")

def main():
    """Run enhanced diagnostics"""
    print("🔍 ENHANCED HYBRID MODE DIAGNOSTIC")
    print("="*80)
    
    # Load config and test case
    config = MemGPTConfig.load()
    
    data_path = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s.json")
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Use first test case
    test_case = full_data[0]
    chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
    
    print(f"Analyzing test case: {test_case['question_id']}")
    print(f"Question: {test_case['question']}")
    print(f"Chat history length: {len(chat_history)} turns")
    
    # Create agent and analyze
    agent = create_test_agent(config, memory_mode="hybrid", beta=0.5)
    analyze_token_based_selection_issue(agent, chat_history)

if __name__ == "__main__":
    main() 