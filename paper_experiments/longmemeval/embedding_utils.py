"""
Embedding Cache Utilities for LongMemEval Benchmark

This module provides utilities to access pre-computed embeddings from the 
embedding cache, allowing benchmark scripts to avoid regenerating embeddings
when running different memory modes.

Usage in benchmark scripts:
    from embedding_utils import EmbeddingCacheManager
    
    cache_manager = EmbeddingCacheManager()
    embeddings = cache_manager.get_embeddings_for_question(question_id, embedding_config)
"""

import os
import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import uuid

# Cache configuration
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "embeddings_cache")

class EmbeddingCacheManager:
    """Manager for accessing cached embeddings in benchmark scripts"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.index_file = self.cache_dir / "embedding_index.json"
        
        # Load index if available
        self.index = self._load_index()
        self._cache_valid = self._validate_cache()
    
    def _load_index(self) -> Dict:
        """Load the embedding index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _validate_cache(self) -> bool:
        """Validate that cache directory and index exist"""
        return (self.cache_dir.exists() and 
                self.embeddings_dir.exists() and 
                self.index_file.exists() and 
                bool(self.index))
    
    def _get_embedding_hash(self, embedding_config) -> str:
        """Generate hash for embedding configuration"""
        import hashlib
        config_str = f"{embedding_config.embedding_model}_{embedding_config.embedding_dim}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def is_cache_available(self) -> bool:
        """Check if cache is available and valid"""
        return self._cache_valid
    
    def get_cache_info(self) -> Dict:
        """Get information about the cache"""
        if not self._cache_valid:
            return {"available": False, "reason": "Cache not found or invalid"}
        
        info = {"available": True, "models": {}}
        for embedding_hash, model_data in self.index.items():
            info["models"][embedding_hash] = {
                "model": model_data.get("model", "unknown"),
                "dim": model_data.get("dim", "unknown"),
                "num_questions": len(model_data.get("questions", {})),
                "questions": list(model_data.get("questions", {}).keys())
            }
        
        return info
    
    def has_embeddings_for_question(self, question_id: str, embedding_config) -> bool:
        """Check if embeddings exist for a specific question"""
        if not self._cache_valid:
            return False
        
        embedding_hash = self._get_embedding_hash(embedding_config)
        return (embedding_hash in self.index and 
                question_id in self.index[embedding_hash].get("questions", {}))
    
    def get_embeddings_for_question(self, question_id: str, embedding_config) -> Optional[List[Tuple]]:
        """
        Load embeddings for a specific question.
        
        Returns:
            List of tuples (user_msg_id, assistant_msg_id, embedding_vector) or None if not found
        """
        if not self._cache_valid:
            return None
        
        embedding_hash = self._get_embedding_hash(embedding_config)
        
        if not self.has_embeddings_for_question(question_id, embedding_config):
            return None
        
        # Get file path
        question_info = self.index[embedding_hash]["questions"][question_id]
        file_path = self.embeddings_dir / question_info["file"]
        
        if not file_path.exists():
            return None
        
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            return data['embeddings']
        
        except Exception:
            return None
    
    def get_available_questions_for_model(self, embedding_config) -> List[str]:
        """Get list of question IDs that have cached embeddings for this model"""
        if not self._cache_valid:
            return []
        
        embedding_hash = self._get_embedding_hash(embedding_config)
        
        if embedding_hash not in self.index:
            return []
        
        return list(self.index[embedding_hash].get("questions", {}).keys())
    
    def get_cache_coverage(self, all_question_ids: List[str], embedding_config) -> Dict:
        """Get coverage statistics for a list of questions"""
        if not self._cache_valid:
            return {
                "total_questions": len(all_question_ids),
                "cached_questions": 0,
                "missing_questions": all_question_ids,
                "coverage_percentage": 0.0
            }
        
        cached_questions = []
        missing_questions = []
        
        for question_id in all_question_ids:
            if self.has_embeddings_for_question(question_id, embedding_config):
                cached_questions.append(question_id)
            else:
                missing_questions.append(question_id)
        
        return {
            "total_questions": len(all_question_ids),
            "cached_questions": len(cached_questions),
            "missing_questions": missing_questions,
            "coverage_percentage": (len(cached_questions) / len(all_question_ids)) * 100 if all_question_ids else 0.0,
            "cached_question_ids": cached_questions
        }

def create_message_id_to_embedding_map(embeddings: List[Tuple]) -> Dict[uuid.UUID, Tuple]:
    """
    Create a mapping from message IDs to their embeddings.
    
    Args:
        embeddings: List of (user_msg_id, assistant_msg_id, embedding_vector)
    
    Returns:
        Dict mapping message_id -> (pair_partner_id, embedding_vector, pair_role)
        where pair_role is 'user' or 'assistant'
    """
    id_to_embedding = {}
    
    for user_msg_id, assistant_msg_id, embedding_vector in embeddings:
        # Map user message ID to its embedding (with assistant partner)
        id_to_embedding[user_msg_id] = (assistant_msg_id, embedding_vector, 'user')
        # Map assistant message ID to its embedding (with user partner)  
        id_to_embedding[assistant_msg_id] = (user_msg_id, embedding_vector, 'assistant')
    
    return id_to_embedding

def find_embeddings_for_message_pairs(embeddings: List[Tuple], target_pairs: List[Tuple[uuid.UUID, uuid.UUID]]) -> List[Tuple]:
    """
    Find embeddings for specific message pairs.
    
    Args:
        embeddings: List of (user_msg_id, assistant_msg_id, embedding_vector)
        target_pairs: List of (user_msg_id, assistant_msg_id) pairs to find
    
    Returns:
        List of embeddings for the target pairs in the same order
    """
    # Create a lookup dictionary
    pair_to_embedding = {}
    for user_id, assistant_id, embedding in embeddings:
        pair_to_embedding[(user_id, assistant_id)] = embedding
    
    # Find embeddings for target pairs
    result_embeddings = []
    for user_id, assistant_id in target_pairs:
        if (user_id, assistant_id) in pair_to_embedding:
            result_embeddings.append((user_id, assistant_id, pair_to_embedding[(user_id, assistant_id)]))
    
    return result_embeddings

# Convenience functions for direct use
def load_cached_embeddings(question_id: str, embedding_config, cache_dir: str = None) -> Optional[List[Tuple]]:
    """
    Simple function to load cached embeddings for a question.
    
    Args:
        question_id: The question ID from the benchmark
        embedding_config: The embedding configuration object
        cache_dir: Optional cache directory path
    
    Returns:
        List of (user_msg_id, assistant_msg_id, embedding_vector) or None if not found
    """
    manager = EmbeddingCacheManager(cache_dir)
    return manager.get_embeddings_for_question(question_id, embedding_config)

def check_embedding_cache_coverage(all_question_ids: List[str], embedding_config, cache_dir: str = None) -> Dict:
    """
    Check what percentage of questions have cached embeddings.
    
    Args:
        all_question_ids: List of all question IDs to check
        embedding_config: The embedding configuration object
        cache_dir: Optional cache directory path
    
    Returns:
        Dictionary with coverage statistics
    """
    manager = EmbeddingCacheManager(cache_dir)
    return manager.get_cache_coverage(all_question_ids, embedding_config)

def is_embedding_cache_available(cache_dir: str = None) -> bool:
    """
    Check if embedding cache is available.
    
    Args:
        cache_dir: Optional cache directory path
    
    Returns:
        True if cache is available and valid
    """
    manager = EmbeddingCacheManager(cache_dir)
    return manager.is_cache_available() 