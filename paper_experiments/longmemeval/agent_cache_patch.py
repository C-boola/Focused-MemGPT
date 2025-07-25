"""
Agent Cache Patch for LongMemEval Benchmark

This module provides patching functions to enhance the Agent class
with cached embedding support. This allows the agent to use pre-computed
embeddings when available, falling back to on-demand generation when needed.

Usage:
    from agent_cache_patch import patch_agent_with_cache
    
    agent = Agent(...)
    patch_agent_with_cache(agent, cache_manager)
"""

import json
from typing import List, Tuple, Optional, Dict, Any
from memgpt.embeddings import create_embedding

def create_cached_message_pair_embeddings(agent, message_sequence, embedding_config, cached_embeddings=None):
    """
    Enhanced embedding function that uses cached embeddings when available.
    Falls back to on-demand generation when cached embeddings are not available.
    
    Args:
        agent: The MemGPT agent instance
        message_sequence: List of Message objects
        embedding_config: Embedding configuration
        cached_embeddings: Pre-computed embeddings (optional)
    
    Returns:
        List of tuples (user_msg_id, assistant_msg_id, embedding_vector)
    """
    
    # If we have cached embeddings, try to use them first
    if cached_embeddings is not None:
        # Verify that cached embeddings match current message sequence
        if _validate_cached_embeddings(message_sequence, cached_embeddings):
            print(f"[CACHE] Using {len(cached_embeddings)} pre-computed embeddings")
            return cached_embeddings
        else:
            print(f"[CACHE] Warning: Cached embeddings don't match current message sequence, generating fresh embeddings")
    
    # Fall back to original embedding generation
    print(f"[CACHE] Generating embeddings on-demand for {len(message_sequence)} messages")
    return _generate_fresh_embeddings(message_sequence, embedding_config)

def _validate_cached_embeddings(message_sequence, cached_embeddings):
    """
    Validate that cached embeddings correspond to the current message sequence.
    
    Args:
        message_sequence: Current message sequence
        cached_embeddings: Cached embeddings to validate
    
    Returns:
        True if cached embeddings are valid for current sequence
    """
    if not cached_embeddings:
        return False
    
    # Create a set of message IDs from current sequence
    current_message_ids = {msg.id for msg in message_sequence if msg.id is not None}
    
    # Check if all cached embedding message IDs exist in current sequence
    cached_message_ids = set()
    for user_id, assistant_id, _ in cached_embeddings:
        cached_message_ids.add(user_id)
        cached_message_ids.add(assistant_id)
    
    # Embeddings are valid if all cached message IDs are present in current sequence
    return cached_message_ids.issubset(current_message_ids)

def _generate_fresh_embeddings(message_sequence, embedding_config):
    """
    Generate fresh embeddings for message pairs.
    This is the same logic as the original _create_robust_message_pair_embeddings.
    """
    pair_embeddings = []
    if not message_sequence or len(message_sequence) < 2:
        return pair_embeddings

    for i in range(len(message_sequence) - 1):
        msg1 = message_sequence[i]
        msg2 = message_sequence[i+1]

        if msg1.role == "user" and msg2.role == "assistant":
            try:
                # Handle both JSON and plain text messages
                text1 = ""
                text2 = ""
                
                # Try JSON parsing first (for regular MemGPT messages)
                try:
                    msg1_content = json.loads(msg1.text) if msg1.text else {}
                    if msg1_content.get('type') == 'user_message':
                        text1 = msg1_content.get('message', '') or msg1.text
                    else:
                        text1 = msg1.text if msg1.text else ""
                except (json.JSONDecodeError, TypeError):
                    # Fall back to plain text (for longmemeval injected messages)
                    text1 = msg1.text if msg1.text else ""
                
                # Assistant messages are typically plain text
                text2 = msg2.text if msg2.text else ""
                
                # Skip empty messages or system-like messages
                if not text1.strip() or not text2.strip():
                    continue
                
                # Skip login messages and other system messages
                if any(keyword in text1.lower() for keyword in ['login', 'bootup', 'system']):
                    continue
                
                # Ensure IDs are not None
                if msg1.id is None or msg2.id is None:
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
                    print(f"[CACHE] Error creating embedding for pair: {e}")
                    continue
                    
            except Exception as e:
                print(f"[CACHE] Error processing message pair at index {i}: {e}")
                continue

    return pair_embeddings

def patch_agent_with_cache(agent, cache_manager=None):
    """
    Patch an agent instance to use cached embeddings when available.
    
    Args:
        agent: Agent instance to patch
        cache_manager: EmbeddingCacheManager instance (optional)
    """
    
    # Store original method
    original_method = agent._create_robust_message_pair_embeddings
    
    def cached_create_robust_message_pair_embeddings(message_sequence, embedding_config):
        """Enhanced method that uses cached embeddings when available"""
        
        # Check if we have cached embeddings stored on the agent
        cached_embeddings = getattr(agent, '_cached_embeddings', None)
        
        # Use the enhanced embedding function
        return create_cached_message_pair_embeddings(
            agent, message_sequence, embedding_config, cached_embeddings
        )
    
    # Replace the method
    agent._create_robust_message_pair_embeddings = cached_create_robust_message_pair_embeddings
    
    # Store reference to cache manager
    agent._cache_manager = cache_manager
    
    print(f"[CACHE] Agent '{agent.agent_state.name}' patched with cache support")

def unpatch_agent_cache(agent):
    """
    Remove cache patching from an agent instance.
    
    Args:
        agent: Agent instance to unpatch
    """
    # This would restore the original method if we stored it
    # For now, we'll just remove the cache attributes
    if hasattr(agent, '_cached_embeddings'):
        delattr(agent, '_cached_embeddings')
    if hasattr(agent, '_cache_manager'):
        delattr(agent, '_cache_manager')
    
    print(f"[CACHE] Cache patching removed from agent '{agent.agent_state.name}'") 