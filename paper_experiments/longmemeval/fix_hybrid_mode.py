#!/usr/bin/env python3
"""
Fix for the FIFO score calculation bug in hybrid mode.
This file contains the corrected logic that should replace the buggy implementation.
"""

def calculate_correct_fifo_scores(self, message_pair_embeddings_with_ids):
    """
    Calculate correct FIFO scores based on actual message positions in the conversation.
    
    Args:
        message_pair_embeddings_with_ids: List of tuples (user_msg_id, assistant_msg_id, embedding)
    
    Returns:
        dict: Mapping from (user_msg_id, assistant_msg_id) to FIFO score
    """
    # Create mapping from message ID to actual position in conversation
    message_id_to_position = {msg.id: idx for idx, msg in enumerate(self._messages)}
    
    pair_fifo_scores = {}
    max_position = len(self._messages) - 1
    
    for user_msg_id, assistant_msg_id, _ in message_pair_embeddings_with_ids:
        # Get actual positions in the conversation
        user_pos = message_id_to_position.get(user_msg_id, 0)
        asst_pos = message_id_to_position.get(assistant_msg_id, 0)
        
        # Use average position of the pair
        avg_position = (user_pos + asst_pos) / 2.0
        
        # Normalize: older messages (lower position) get higher scores for removal
        if max_position > 0:
            fifo_score = (max_position - avg_position) / max_position
        else:
            fifo_score = 0.0
        
        # Ensure score is between 0 and 1
        fifo_score = max(0.0, min(1.0, fifo_score))
        
        pair_fifo_scores[(user_msg_id, assistant_msg_id)] = fifo_score
    
    return pair_fifo_scores

# CORRECTED hybrid summarization logic (replace the buggy section)
def summarize_messages_hybrid_inplace_FIXED(self):
    """
    FIXED version of hybrid summarization with correct FIFO score calculation.
    """
    print("HYBRID MODE: Summarizing messages using hybrid approach (FIFO + Focus).")
    
    # ... [previous code remains the same until FIFO score calculation] ...
    
    # Step 3: Create FIFO scores (FIXED VERSION)
    print("Hybrid summarize: Calculating correct FIFO scores based on actual message positions...")
    
    # Create mapping from message ID to actual position in conversation
    message_id_to_position = {msg.id: idx for idx, msg in enumerate(self._messages)}
    
    pair_fifo_scores = {}
    max_position = len(self._messages) - 1
    
    for user_msg_id, assistant_msg_id, _ in message_pair_embeddings_with_ids:
        # Get actual positions in the conversation
        user_pos = message_id_to_position.get(user_msg_id, 0)
        asst_pos = message_id_to_position.get(assistant_msg_id, 0)
        
        # Use average position of the pair
        avg_position = (user_pos + asst_pos) / 2.0
        
        # Normalize: older messages (lower position) get higher scores for removal
        if max_position > 0:
            fifo_score = (max_position - avg_position) / max_position
        else:
            fifo_score = 0.0
        
        # Ensure score is between 0 and 1
        fifo_score = max(0.0, min(1.0, fifo_score))
        
        pair_fifo_scores[(user_msg_id, assistant_msg_id)] = fifo_score
    
    print(f"Hybrid summarize: Calculated FIFO scores for {len(pair_fifo_scores)} pairs")
    
    # ... [rest of the code remains the same] ...

# Alternative approach: Use timestamps if available
def calculate_timestamp_based_fifo_scores(self, message_pair_embeddings_with_ids):
    """
    Calculate FIFO scores based on message timestamps (more accurate than positions).
    """
    pair_fifo_scores = {}
    
    # Get all timestamps for normalization
    all_timestamps = []
    for msg in self._messages:
        if msg.created_at:
            all_timestamps.append(msg.created_at.timestamp())
    
    if not all_timestamps:
        # Fallback to position-based scoring
        return self.calculate_correct_fifo_scores(message_pair_embeddings_with_ids)
    
    min_timestamp = min(all_timestamps)
    max_timestamp = max(all_timestamps)
    timestamp_range = max_timestamp - min_timestamp
    
    for user_msg_id, assistant_msg_id, _ in message_pair_embeddings_with_ids:
        # Find the messages
        user_msg = next((msg for msg in self._messages if msg.id == user_msg_id), None)
        asst_msg = next((msg for msg in self._messages if msg.id == assistant_msg_id), None)
        
        if not user_msg or not asst_msg or not user_msg.created_at or not asst_msg.created_at:
            # Fallback to 0.5 if timestamps are missing
            pair_fifo_scores[(user_msg_id, assistant_msg_id)] = 0.5
            continue
        
        # Use average timestamp of the pair
        avg_timestamp = (user_msg.created_at.timestamp() + asst_msg.created_at.timestamp()) / 2.0
        
        # Normalize: older messages (lower timestamp) get higher scores for removal
        if timestamp_range > 0:
            fifo_score = (max_timestamp - avg_timestamp) / timestamp_range
        else:
            fifo_score = 0.5  # All messages have same timestamp
        
        # Ensure score is between 0 and 1
        fifo_score = max(0.0, min(1.0, fifo_score))
        
        pair_fifo_scores[(user_msg_id, assistant_msg_id)] = fifo_score
    
    return pair_fifo_scores

if __name__ == "__main__":
    print("This file contains fixes for the hybrid mode FIFO scoring bug.")
    print("To apply the fix:")
    print("1. Run the diagnostic script first: python diagnose_hybrid_mode.py")
    print("2. Replace the FIFO scoring section in agent.py with the fixed version above")
    print("3. Test with different beta values to confirm they now produce different results") 