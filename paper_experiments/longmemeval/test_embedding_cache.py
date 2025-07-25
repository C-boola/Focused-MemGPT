#!/usr/bin/env python3
"""
Test Script for LongMemEval Embedding Cache System

This script validates that the embedding cache system works correctly by:
1. Testing cache creation and storage
2. Testing cache retrieval and validation
3. Testing cache coverage analysis
4. Testing cache management functions

Usage:
    python test_embedding_cache.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import uuid

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from embedding_utils import EmbeddingCacheManager, check_embedding_cache_coverage
from memgpt.config import MemGPTConfig

def test_cache_basic_operations():
    """Test basic cache operations"""
    print("=" * 50)
    print("Testing Basic Cache Operations")
    print("=" * 50)
    
    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp(prefix="test_cache_")
    print(f"Using temporary cache directory: {temp_dir}")
    
    try:
        # Initialize cache manager
        cache = EmbeddingCacheManager(temp_dir)
        
        # Check initial state
        assert not cache.is_cache_available(), "Cache should not be available initially"
        print("✅ Initial cache state correct")
        
        # Load config for embedding config
        config = MemGPTConfig.load()
        embedding_config = config.default_embedding_config
        
        # Create mock embeddings
        mock_embeddings = [
            (uuid.uuid4(), uuid.uuid4(), [0.1, 0.2, 0.3] * 100),  # 300D vector
            (uuid.uuid4(), uuid.uuid4(), [0.4, 0.5, 0.6] * 100),
            (uuid.uuid4(), uuid.uuid4(), [0.7, 0.8, 0.9] * 100),
        ]
        
        # Store embeddings
        question_id = "test_question_001"
        metadata = {"test": True, "num_sessions": 3}
        
        cache.store_embeddings(
            question_id=question_id,
            embeddings=mock_embeddings,
            embedding_config=embedding_config,
            metadata=metadata
        )
        print("✅ Embeddings stored successfully")
        
        # Check cache is now available
        assert cache.is_cache_available(), "Cache should be available after storing embeddings"
        print("✅ Cache availability correct after storage")
        
        # Retrieve embeddings
        retrieved_embeddings = cache.load_embeddings(question_id, embedding_config)
        assert retrieved_embeddings is not None, "Should be able to retrieve stored embeddings"
        assert len(retrieved_embeddings) == len(mock_embeddings), "Retrieved embeddings should match stored count"
        print("✅ Embeddings retrieved successfully")
        
        # Validate embedding content
        for i, (original, retrieved) in enumerate(zip(mock_embeddings, retrieved_embeddings)):
            assert original[0] == retrieved[0], f"User ID mismatch at index {i}"
            assert original[1] == retrieved[1], f"Assistant ID mismatch at index {i}"
            assert original[2] == retrieved[2], f"Embedding vector mismatch at index {i}"
        print("✅ Embedding content validation passed")
        
        # Test has_embeddings
        assert cache.has_embeddings_for_question(question_id, embedding_config), "Should report embeddings exist"
        assert not cache.has_embeddings_for_question("nonexistent", embedding_config), "Should report no embeddings for nonexistent question"
        print("✅ has_embeddings_for_question working correctly")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    print("✅ All basic cache operations tests passed!\n")

def test_cache_coverage_analysis():
    """Test cache coverage analysis"""
    print("=" * 50)
    print("Testing Cache Coverage Analysis")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp(prefix="test_coverage_")
    print(f"Using temporary cache directory: {temp_dir}")
    
    try:
        cache = EmbeddingCacheManager(temp_dir)
        config = MemGPTConfig.load()
        embedding_config = config.default_embedding_config
        
        # Create test questions
        all_questions = ["q1", "q2", "q3", "q4", "q5"]
        cached_questions = ["q1", "q3", "q5"]  # Cache only some questions
        
        # Store embeddings for some questions
        for q_id in cached_questions:
            mock_embeddings = [
                (uuid.uuid4(), uuid.uuid4(), [0.1, 0.2, 0.3] * 100)
            ]
            cache.store_embeddings(
                question_id=q_id,
                embeddings=mock_embeddings,
                embedding_config=embedding_config
            )
        
        print(f"Stored embeddings for: {cached_questions}")
        
        # Test coverage analysis
        coverage = cache.get_cache_coverage(all_questions, embedding_config)
        
        assert coverage["total_questions"] == len(all_questions), "Total questions count incorrect"
        assert coverage["cached_questions"] == len(cached_questions), "Cached questions count incorrect"
        assert set(coverage["missing_questions"]) == {"q2", "q4"}, "Missing questions list incorrect"
        assert coverage["coverage_percentage"] == 60.0, "Coverage percentage incorrect"
        
        print(f"✅ Coverage analysis correct: {coverage['coverage_percentage']:.1f}%")
        
        # Test convenience function
        coverage2 = check_embedding_cache_coverage(all_questions, embedding_config, temp_dir)
        assert coverage == coverage2, "Convenience function should return same result"
        print("✅ Convenience function working correctly")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    print("✅ All coverage analysis tests passed!\n")

def test_cache_info_and_stats():
    """Test cache info and statistics"""
    print("=" * 50)
    print("Testing Cache Info and Statistics")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp(prefix="test_stats_")
    print(f"Using temporary cache directory: {temp_dir}")
    
    try:
        cache = EmbeddingCacheManager(temp_dir)
        config = MemGPTConfig.load()
        embedding_config = config.default_embedding_config
        
        # Initially no cache
        info = cache.get_cache_info()
        assert not info["available"], "Cache should not be available initially"
        print("✅ Initial cache info correct")
        
        # Add some embeddings
        questions = ["test_q1", "test_q2", "test_q3"]
        for q_id in questions:
            mock_embeddings = [
                (uuid.uuid4(), uuid.uuid4(), [0.1] * 100),
                (uuid.uuid4(), uuid.uuid4(), [0.2] * 100)
            ]
            cache.store_embeddings(q_id, mock_embeddings, embedding_config)
        
        # Check updated info
        info = cache.get_cache_info()
        assert info["available"], "Cache should be available after adding embeddings"
        assert len(info["models"]) == 1, "Should have one embedding model"
        
        model_hash = list(info["models"].keys())[0]
        model_info = info["models"][model_hash]
        
        assert model_info["num_questions"] == len(questions), "Should track correct number of questions"
        assert set(model_info["questions"]) == set(questions), "Should track correct question IDs"
        
        print(f"✅ Cache info correct: {model_info['num_questions']} questions for model {model_info['model']}")
        
        # Test available questions
        available_questions = cache.get_available_questions_for_model(embedding_config)
        assert set(available_questions) == set(questions), "Available questions should match stored questions"
        print("✅ Available questions list correct")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    print("✅ All cache info and statistics tests passed!\n")

def test_cache_file_structure():
    """Test cache file structure and organization"""
    print("=" * 50)
    print("Testing Cache File Structure")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp(prefix="test_structure_")
    print(f"Using temporary cache directory: {temp_dir}")
    
    try:
        cache = EmbeddingCacheManager(temp_dir)
        config = MemGPTConfig.load()
        embedding_config = config.default_embedding_config
        
        # Check directory structure creation
        cache_path = Path(temp_dir)
        embeddings_dir = cache_path / "embeddings"
        metadata_dir = cache_path / "metadata"
        
        assert embeddings_dir.exists(), "Embeddings directory should be created"
        assert metadata_dir.exists(), "Metadata directory should be created"
        print("✅ Directory structure created correctly")
        
        # Store an embedding and check file creation
        question_id = "structure_test"
        mock_embeddings = [(uuid.uuid4(), uuid.uuid4(), [0.1] * 100)]
        
        cache.store_embeddings(question_id, mock_embeddings, embedding_config)
        
        # Check index file exists
        index_file = cache_path / "embedding_index.json"
        assert index_file.exists(), "Index file should be created"
        
        # Check embedding file exists
        embedding_files = list(embeddings_dir.glob("*.pkl.gz"))
        assert len(embedding_files) == 1, "Should have one embedding file"
        assert question_id in embedding_files[0].name, "Embedding file should contain question ID"
        print("✅ Files created with correct naming")
        
        # Check index content
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        assert len(index_data) == 1, "Index should have one model entry"
        model_hash = list(index_data.keys())[0]
        assert question_id in index_data[model_hash]["questions"], "Question should be in index"
        print("✅ Index file content correct")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    print("✅ All file structure tests passed!\n")

def test_error_handling():
    """Test error handling and edge cases"""
    print("=" * 50)
    print("Testing Error Handling and Edge Cases")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp(prefix="test_errors_")
    print(f"Using temporary cache directory: {temp_dir}")
    
    try:
        cache = EmbeddingCacheManager(temp_dir)
        config = MemGPTConfig.load()
        embedding_config = config.default_embedding_config
        
        # Test loading non-existent embeddings
        result = cache.load_embeddings("nonexistent_question", embedding_config)
        assert result is None, "Should return None for non-existent embeddings"
        print("✅ Non-existent embedding handling correct")
        
        # Test empty embeddings list
        cache.store_embeddings("empty_test", [], embedding_config)
        result = cache.load_embeddings("empty_test", embedding_config)
        assert result == [], "Should handle empty embeddings list"
        print("✅ Empty embeddings list handling correct")
        
        # Test invalid cache directory (read-only)
        readonly_dir = tempfile.mkdtemp(prefix="readonly_")
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        try:
            readonly_cache = EmbeddingCacheManager(readonly_dir)
            # This should not crash, but cache won't be available
            assert not readonly_cache.is_cache_available(), "Read-only cache should not be available"
            print("✅ Read-only directory handling correct")
        finally:
            os.chmod(readonly_dir, 0o755)  # Restore permissions for cleanup
            shutil.rmtree(readonly_dir)
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    print("✅ All error handling tests passed!\n")

def main():
    """Run all tests"""
    print("🧪 Starting LongMemEval Embedding Cache System Tests")
    print("=" * 70)
    
    try:
        # Check if MemGPT config is available
        config = MemGPTConfig.load()
        print(f"✅ MemGPT config loaded successfully")
        print(f"   Embedding model: {config.default_embedding_config.embedding_model}")
        print(f"   Embedding dim: {config.default_embedding_config.embedding_dim}")
        print()
        
        # Run all tests
        test_cache_basic_operations()
        test_cache_coverage_analysis()
        test_cache_info_and_stats() 
        test_cache_file_structure()
        test_error_handling()
        
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("The embedding cache system is working correctly.")
        print("You can now use it with confidence in your benchmark runs.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 