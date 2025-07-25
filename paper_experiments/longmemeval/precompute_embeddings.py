#!/usr/bin/env python3
"""
Embedding Precomputation Script for LongMemEval Benchmark

This script pre-generates vector embeddings for all message pairs in the LongMemEval
benchmark dataset and stores them in an organized, compressed format. This allows
benchmark runs using embedding-based memory modes (focus, hybrid, density) to load
pre-computed embeddings instead of generating them on-the-fly, saving significant
time and API costs.

Features:
- Generates embeddings for all user-assistant message pairs
- Organized storage with question_id mapping
- Compression and metadata tracking
- Utilities for loading embeddings during benchmark runs
- Progress tracking and resume capability
- Embedding model/version tracking for consistency
"""

import json
import os
import uuid
import time
import sys
import traceback
import pickle
import gzip
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock
import argparse

# MemGPT imports
from memgpt.agent import Agent
from memgpt.interface import AgentInterface
from memgpt.data_types import AgentState
from memgpt.presets.presets import available_presets
from memgpt.config import MemGPTConfig
from memgpt.constants import DEFAULT_PRESET, DEFAULT_PERSONA, DEFAULT_HUMAN
from memgpt.utils import get_persona_text, get_human_text
from memgpt.prompts import gpt_system
from memgpt.presets.presets import generate_functions_json
from memgpt.embeddings import create_embedding

# Configuration
DEBUG = True
MODULE_BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_s.json")
ORACLE_PATH = os.path.join(MODULE_BASE_PATH, "data", "longmemeval_oracle.json")
EMBEDDINGS_CACHE_DIR = os.path.join(MODULE_BASE_PATH, "embeddings_cache")

FOCUSED_QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference", 
    "multi-session"
}

# Global lock for thread-safe cache operations
cache_lock = Lock()

def log_debug(message):
    """Simple logging function"""
    if DEBUG:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}")

class SilentInterface(AgentInterface):
    """Silent interface for embedding generation (no output)"""
    def user_message(self, msg, msg_obj=None) -> None: pass
    def internal_monologue(self, msg, msg_obj=None) -> None: pass  
    def assistant_message(self, msg, msg_obj=None) -> None: pass
    def function_message(self, msg, msg_obj=None) -> None: pass

class EmbeddingCache:
    """Manages storage and retrieval of pre-computed embeddings"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.metadata_dir = self.cache_dir / "metadata"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.index_file = self.cache_dir / "embedding_index.json"
        self.config_file = self.cache_dir / "cache_config.json"
        
        # Load or create index
        self.index = self._load_index()
        self.config = self._load_config()
    
    def _load_index(self) -> Dict:
        """Load the embedding index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save the embedding index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _load_config(self) -> Dict:
        """Load cache configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_config(self, config: Dict):
        """Save cache configuration"""
        self.config = config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_embedding_hash(self, embedding_config) -> str:
        """Generate hash for embedding configuration"""
        config_str = f"{embedding_config.embedding_model}_{embedding_config.embedding_dim}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_question_file(self, question_id: str, embedding_hash: str) -> Path:
        """Get the file path for a question's embeddings"""
        return self.embeddings_dir / f"{question_id}_{embedding_hash}.pkl.gz"
    
    def store_embeddings(self, question_id: str, embeddings: List[Tuple], embedding_config, metadata: Dict = None):
        """Store embeddings for a question (thread-safe)"""
        embedding_hash = self._get_embedding_hash(embedding_config)
        file_path = self._get_question_file(question_id, embedding_hash)
        
        # Prepare data for storage
        embedding_data = {
            'question_id': question_id,
            'embeddings': embeddings,  # List of (user_msg_id, assistant_msg_id, embedding_vector)
            'embedding_config': {
                'model': embedding_config.embedding_model,
                'dim': embedding_config.embedding_dim,
                'hash': embedding_hash
            },
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'num_pairs': len(embeddings)
        }
        
        # Save compressed
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(embedding_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Thread-safe index update
        with cache_lock:
            # Reload index to get latest state
            self.index = self._load_index()
            
            # Update index
            if embedding_hash not in self.index:
                self.index[embedding_hash] = {
                    'model': embedding_config.embedding_model,
                    'dim': embedding_config.embedding_dim,
                    'questions': {}
                }
            
            self.index[embedding_hash]['questions'][question_id] = {
                'file': str(file_path.name),
                'num_pairs': len(embeddings),
                'created_at': embedding_data['created_at']
            }
            
            self._save_index()
        
        log_debug(f"Stored {len(embeddings)} embeddings for question {question_id}")
    
    def load_embeddings(self, question_id: str, embedding_config) -> Optional[List[Tuple]]:
        """Load embeddings for a question"""
        embedding_hash = self._get_embedding_hash(embedding_config)
        
        if embedding_hash not in self.index:
            return None
        
        if question_id not in self.index[embedding_hash]['questions']:
            return None
        
        file_path = self._get_question_file(question_id, embedding_hash)
        
        if not file_path.exists():
            log_debug(f"Warning: Index entry exists but file missing for {question_id}")
            return None
        
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            log_debug(f"Loaded {len(data['embeddings'])} embeddings for question {question_id}")
            return data['embeddings']
        
        except Exception as e:
            log_debug(f"Error loading embeddings for {question_id}: {e}")
            return None
    
    def has_embeddings(self, question_id: str, embedding_config) -> bool:
        """Check if embeddings exist for a question"""
        embedding_hash = self._get_embedding_hash(embedding_config)
        return (embedding_hash in self.index and 
                question_id in self.index[embedding_hash]['questions'])
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_models': len(self.index),
            'models': {}
        }
        
        for embedding_hash, model_data in self.index.items():
            stats['models'][embedding_hash] = {
                'model': model_data['model'],
                'dim': model_data['dim'],
                'num_questions': len(model_data['questions']),
                'questions': list(model_data['questions'].keys())
            }
        
        return stats
    
    def cleanup_orphaned_files(self):
        """Remove files not referenced in index"""
        referenced_files = set()
        for model_data in self.index.values():
            for question_data in model_data['questions'].values():
                referenced_files.add(question_data['file'])
        
        existing_files = list(self.embeddings_dir.glob("*.pkl.gz"))
        orphaned = [f for f in existing_files if f.name not in referenced_files]
        
        for orphan in orphaned:
            log_debug(f"Removing orphaned file: {orphan}")
            orphan.unlink()
        
        return len(orphaned)

def load_and_filter_data() -> List[Dict]:
    """Load and filter LongMemEval data"""
    log_debug("Loading and filtering benchmark data...")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    with open(ORACLE_PATH, 'r', encoding='utf-8') as f:
        oracle_list = json.load(f)
    
    oracle_data = {item['question_id']: item for item in oracle_list}
    
    filtered_data = [
        item for item in full_data
        if oracle_data.get(item['question_id'], {}).get('question_type') in FOCUSED_QUESTION_TYPES
    ]
    
    log_debug(f"Loaded {len(filtered_data)} test cases (filtered from {len(full_data)} total)")
    return filtered_data

def create_robust_message_pair_embeddings(message_sequence, embedding_config):
    """
    Create embeddings for message pairs, handling both JSON and plain text.
    This is a standalone version of the Agent method.
    """
    import json
    
    pair_embeddings = []
    if not message_sequence or len(message_sequence) < 2:
        return pair_embeddings

    log_debug(f"Creating embeddings for {len(message_sequence)} messages...")
    
    for i in range(len(message_sequence) - 1):
        msg1 = message_sequence[i]
        msg2 = message_sequence[i+1]

        # Look for user-assistant pairs
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
                    log_debug(f"Warning: Skipping message pair due to missing ID(s)")
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
                    log_debug(f"Error creating embedding for pair: {e}")
                    continue
                    
            except Exception as e:
                log_debug(f"Error processing message pair at index {i}: {e}")
                continue

    log_debug(f"Successfully created {len(pair_embeddings)} message pair embeddings")
    return pair_embeddings

def run_single_embedding_generation(args_tuple):
    """
    Worker function to generate embeddings for a single test case.
    Takes a tuple of arguments to work with multiprocessing.
    """
    (test_case, config_dict, cache_dir, worker_id) = args_tuple
    q_id = test_case['question_id']
    
    try:
        # Recreate config object from dict (needed for multiprocessing)
        config = MemGPTConfig(**config_dict)
        
        # Create cache instance for this worker
        cache = EmbeddingCache(cache_dir)
        
        # Generate embeddings
        success = generate_embeddings_for_question_worker(config, test_case, cache, worker_id)
        
        return {
            "success": success,
            "question_id": q_id,
            "worker_id": worker_id
        }
        
    except Exception as e:
        error_msg = f"ERROR in worker {worker_id} for case {q_id}: {str(e)}"
        return {
            "success": False,
            "question_id": q_id,
            "worker_id": worker_id,
            "error": error_msg
        }

def generate_embeddings_for_question_worker(base_config: MemGPTConfig, test_case: Dict, cache: EmbeddingCache, worker_id: int) -> bool:
    """Worker version of generate_embeddings_for_question that runs in a separate process"""
    q_id = test_case['question_id']
    
    # Check if embeddings already exist
    if cache.has_embeddings(q_id, base_config.default_embedding_config):
        log_debug(f"Worker {worker_id}: Embeddings already exist for question {q_id}, skipping")
        return True
    
    log_debug(f"Worker {worker_id}: Generating embeddings for question {q_id}")
    
    try:
        # Create a temporary agent for embedding generation
        dummy_user_id = uuid.uuid4()
        preset_config = available_presets[DEFAULT_PRESET]
        preset_system_prompt = preset_config["system_prompt"]
        preset_function_set_names = preset_config["functions"]
        functions_schema = generate_functions_json(preset_function_set_names)
        
        agent_state = AgentState(
            name=f"embedding_agent_{q_id}_w{worker_id}",
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
                "mem_mode": "focus",  # Doesn't matter for embedding generation
                "beta": 0.5,
                "cluster_summaries": False,
                "centroid_method": "centroid",
                "score_mode": None,
                "max_tokens": None,
            },
        )

        agent = Agent(
            interface=SilentInterface(), 
            agent_state=agent_state,
            mem_mode="focus"
        )
        
        # Inject conversation history
        full_chat_history = [turn for session in test_case['haystack_sessions'] for turn in session]
        agent.append_to_messages(full_chat_history)
        
        log_debug(f"Worker {worker_id}: Injected {len(full_chat_history)} messages into agent")
        
        # Generate embeddings
        embeddings = create_robust_message_pair_embeddings(
            message_sequence=agent._messages,
            embedding_config=base_config.default_embedding_config
        )
        
        if not embeddings:
            log_debug(f"Worker {worker_id}: No embeddings generated for question {q_id}")
            return False
        
        # Store embeddings with metadata
        metadata = {
            'question': test_case['question'],
            'num_sessions': len(test_case['haystack_sessions']),
            'num_turns': len(full_chat_history),
            'num_agent_messages': len(agent._messages),
            'worker_id': worker_id
        }
        
        cache.store_embeddings(
            question_id=q_id,
            embeddings=embeddings,
            embedding_config=base_config.default_embedding_config,
            metadata=metadata
        )
        
        log_debug(f"Worker {worker_id}: Successfully generated and cached {len(embeddings)} embeddings for question {q_id}")
        return True
        
    except Exception as e:
        log_debug(f"Worker {worker_id}: Error generating embeddings for question {q_id}: {e}")
        traceback.print_exc()
        return False



def main():
    """Main function to precompute embeddings with parallel processing"""
    print("=" * 70)
    print("  LongMemEval Embedding Precomputation Script (Parallel)")
    print("=" * 70)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Precompute embeddings for LongMemEval benchmark with parallel processing")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limited cases)")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing embeddings")
    parser.add_argument("--cleanup", action="store_true", help="Clean up orphaned cache files")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of test cases to process in parallel")
    parser.add_argument("--max-workers", type=int, help="Maximum number of worker processes (default: batch-size)")
    
    args = parser.parse_args()
    
    if args.max_workers is None:
        args.max_workers = args.batch_size
    
    # Initialize cache
    cache = EmbeddingCache(EMBEDDINGS_CACHE_DIR)
    
    if args.cleanup:
        print("Cleaning up orphaned files...")
        removed = cache.cleanup_orphaned_files()
        print(f"Removed {removed} orphaned files")
        return
    
    # Load configuration
    config = MemGPTConfig.load()
    config_dict = config.__dict__  # Convert to dict for multiprocessing
    
    print(f"Using embedding model: {config.default_embedding_config.embedding_model}")
    print(f"Cache directory: {EMBEDDINGS_CACHE_DIR}")
    
    # Load test data
    test_cases = load_and_filter_data()
    
    if args.test:
        test_cases = test_cases[:25]  # Limit for testing
        print(f"Test mode: Processing {len(test_cases)} cases")
    else:
        print(f"Processing {len(test_cases)} test cases")
    
    # Check existing cache
    if not args.force:
        cache_stats = cache.get_cache_stats()
        if cache_stats['total_models'] > 0:
            print(f"\nExisting cache found:")
            for model_hash, model_info in cache_stats['models'].items():
                print(f"  Model {model_info['model']}: {model_info['num_questions']} questions cached")
    
    print(f"\nConfiguration:")
    print(f"  Force regenerate: {'ON' if args.force else 'OFF'}")
    print(f"  Test mode: {'ON' if args.test else 'OFF'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max workers: {args.max_workers}")
    
    # Filter out already processed cases if not forcing regeneration
    remaining_test_cases = []
    skipped = 0
    
    for test_case in test_cases:
        q_id = test_case['question_id']
        if not args.force and cache.has_embeddings(q_id, config.default_embedding_config):
            skipped += 1
        else:
            remaining_test_cases.append(test_case)
    
    if skipped > 0:
        print(f"  Skipping {skipped} already processed cases")
    
    if not remaining_test_cases:
        print("No remaining test cases to process!")
        return
    
    print(f"  Processing {len(remaining_test_cases)} remaining cases")
    
    # Prepare arguments for workers
    worker_args = []
    for i, test_case in enumerate(remaining_test_cases):
        worker_id = i % args.max_workers
        args_tuple = (test_case, config_dict, EMBEDDINGS_CACHE_DIR, worker_id)
        worker_args.append(args_tuple)
    
    # Process in batches
    start_time = time.time()
    successful = 0
    failed = 0
    
    print(f"\nStarting parallel embedding generation...")
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Process in batches
            for batch_start in tqdm(range(0, len(worker_args), args.batch_size), desc="Processing batches"):
                batch_end = min(batch_start + args.batch_size, len(worker_args))
                batch_args = worker_args[batch_start:batch_end]
                
                # Submit batch jobs
                future_to_args = {executor.submit(run_single_embedding_generation, args): args for args in batch_args}
                
                # Collect results for this batch
                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        
                        if result["success"]:
                            successful += 1
                        else:
                            failed += 1
                            if "error" in result:
                                print(f"Error in case {result['question_id']}: {result.get('error', 'Unknown error')}")
                                
                    except Exception as e:
                        failed += 1
                        print(f"Future execution error: {e}")
                
                # Progress update
                batch_num = batch_start // args.batch_size + 1
                total_batches = (len(worker_args) + args.batch_size - 1) // args.batch_size
                print(f"Completed batch {batch_num}/{total_batches}. "
                      f"Total processed: {successful + failed}, Successful: {successful}, Failed: {failed}")
                      
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("  EMBEDDING PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    # Final cache stats
    final_stats = cache.get_cache_stats()
    print(f"\nFinal cache statistics:")
    for model_hash, model_info in final_stats['models'].items():
        print(f"  Model {model_info['model']}: {model_info['num_questions']} questions cached")
    
    print(f"\nCache directory: {EMBEDDINGS_CACHE_DIR}")
    print("Use load_cached_embeddings() in your benchmark scripts to access these embeddings.")

# Utility functions for benchmark scripts
def load_cached_embeddings(question_id: str, embedding_config, cache_dir: str = None) -> Optional[List[Tuple]]:
    """Utility function to load cached embeddings from benchmark scripts"""
    if cache_dir is None:
        cache_dir = EMBEDDINGS_CACHE_DIR
    
    cache = EmbeddingCache(cache_dir)
    return cache.load_embeddings(question_id, embedding_config)

def has_cached_embeddings(question_id: str, embedding_config, cache_dir: str = None) -> bool:
    """Utility function to check if cached embeddings exist"""
    if cache_dir is None:
        cache_dir = EMBEDDINGS_CACHE_DIR
    
    cache = EmbeddingCache(cache_dir)
    return cache.has_embeddings(question_id, embedding_config)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    main() 