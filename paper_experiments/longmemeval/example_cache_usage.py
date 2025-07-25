#!/usr/bin/env python3
"""
Example Usage of LongMemEval Embedding Cache System

This script demonstrates how to use the embedding cache system
in your own scripts or experiments.

Usage:
    python example_cache_usage.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from embedding_utils import (
    EmbeddingCacheManager, 
    load_cached_embeddings, 
    check_embedding_cache_coverage,
    is_embedding_cache_available
)
from memgpt.config import MemGPTConfig

def example_basic_usage():
    """Example of basic cache usage"""
    print("🚀 Example 1: Basic Cache Usage")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = EmbeddingCacheManager()
    
    # Check if cache is available
    if cache_manager.is_cache_available():
        print("✅ Embedding cache is available!")
        
        # Get cache information
        cache_info = cache_manager.get_cache_info()
        print(f"📊 Cache contains {len(cache_info['models'])} embedding models:")
        
        for model_hash, model_info in cache_info['models'].items():
            print(f"   • {model_info['model']}: {model_info['num_questions']} questions")
            
            # Show first few question IDs as examples
            questions = model_info['questions'][:5]  # First 5
            if questions:
                print(f"     Sample questions: {', '.join(questions)}")
                if len(model_info['questions']) > 5:
                    print(f"     ... and {len(model_info['questions']) - 5} more")
    else:
        print("❌ Embedding cache not available")
        print("   Run 'python precompute_embeddings.py' to build the cache")
        return
    
    print()

def example_load_specific_embeddings():
    """Example of loading embeddings for a specific question"""
    print("🔍 Example 2: Loading Specific Embeddings")
    print("=" * 50)
    
    # Load MemGPT config
    config = MemGPTConfig.load()
    
    # Initialize cache manager
    cache_manager = EmbeddingCacheManager()
    
    if not cache_manager.is_cache_available():
        print("❌ Cache not available for this example")
        return
    
    # Get available questions
    available_questions = cache_manager.get_available_questions_for_model(
        config.default_embedding_config
    )
    
    if not available_questions:
        print("❌ No questions available in cache")
        return
    
    # Pick the first available question as an example
    example_question_id = available_questions[0]
    print(f"📄 Loading embeddings for question: {example_question_id}")
    
    # Load embeddings using cache manager
    embeddings = cache_manager.get_embeddings_for_question(
        example_question_id, 
        config.default_embedding_config
    )
    
    if embeddings:
        print(f"✅ Loaded {len(embeddings)} message pair embeddings")
        print(f"   Each embedding is {len(embeddings[0][2])}D")
        
        # Show first few embeddings
        for i, (user_id, assistant_id, vector) in enumerate(embeddings[:3]):
            print(f"   Pair {i+1}: User {str(user_id)[:8]}... + Assistant {str(assistant_id)[:8]}...")
        
        if len(embeddings) > 3:
            print(f"   ... and {len(embeddings) - 3} more pairs")
    else:
        print("❌ No embeddings found for this question")
    
    print()

def example_coverage_analysis():
    """Example of analyzing cache coverage for your dataset"""
    print("📊 Example 3: Cache Coverage Analysis")
    print("=" * 50)
    
    # Load data (simulating what your script might do)
    from precompute_embeddings import load_and_filter_data
    
    try:
        test_cases = load_and_filter_data()
        all_question_ids = [case['question_id'] for case in test_cases]
        print(f"📋 Analyzing coverage for {len(all_question_ids)} questions from LongMemEval dataset")
    except Exception as e:
        print(f"❌ Could not load LongMemEval data: {e}")
        print("   This example requires the LongMemEval dataset to be available")
        return
    
    # Load config
    config = MemGPTConfig.load()
    
    # Check coverage
    coverage = check_embedding_cache_coverage(
        all_question_ids, 
        config.default_embedding_config
    )
    
    print(f"📈 Cache Coverage Results:")
    print(f"   Total questions: {coverage['total_questions']}")
    print(f"   Cached questions: {coverage['cached_questions']}")
    print(f"   Missing questions: {len(coverage['missing_questions'])}")
    print(f"   Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    
    if coverage['coverage_percentage'] == 100:
        print("🎉 Perfect coverage! All questions have cached embeddings.")
    elif coverage['coverage_percentage'] > 80:
        print("🟢 Good coverage! Most questions have cached embeddings.")
    elif coverage['coverage_percentage'] > 50:
        print("🟡 Partial coverage. Consider running precompute_embeddings.py to improve coverage.")
    else:
        print("🔴 Low coverage. Run precompute_embeddings.py to generate cache.")
    
    # Show sample missing questions
    if coverage['missing_questions']:
        print(f"   Missing examples: {', '.join(coverage['missing_questions'][:5])}")
        if len(coverage['missing_questions']) > 5:
            print(f"   ... and {len(coverage['missing_questions']) - 5} more")
    
    print()

def example_convenience_functions():
    """Example of using convenience functions"""
    print("🛠️  Example 4: Convenience Functions")
    print("=" * 50)
    
    # Simple cache availability check
    is_available = is_embedding_cache_available()
    print(f"Cache available: {is_available}")
    
    if not is_available:
        print("❌ Cache not available for convenience function examples")
        return
    
    # Load config
    config = MemGPTConfig.load()
    
    # Get available questions
    cache_manager = EmbeddingCacheManager()
    available_questions = cache_manager.get_available_questions_for_model(
        config.default_embedding_config
    )
    
    if not available_questions:
        print("❌ No questions available for convenience function examples")
        return
    
    # Use convenience function to load embeddings
    example_question = available_questions[0]
    print(f"📄 Using convenience function to load: {example_question}")
    
    embeddings = load_cached_embeddings(
        example_question, 
        config.default_embedding_config
    )
    
    if embeddings:
        print(f"✅ Convenience function loaded {len(embeddings)} embeddings")
    else:
        print("❌ Convenience function returned no embeddings")
    
    print()

def example_integration_pattern():
    """Example of typical integration pattern in your own scripts"""
    print("🔧 Example 5: Integration Pattern for Your Scripts")
    print("=" * 50)
    
    print("Here's how you would typically integrate the cache in your own scripts:")
    print()
    
    example_code = '''
# At the top of your script
from embedding_utils import EmbeddingCacheManager, is_embedding_cache_available
from memgpt.config import MemGPTConfig

def run_your_experiment():
    # Load config
    config = MemGPTConfig.load()
    
    # Check if cache is available
    cache_manager = EmbeddingCacheManager()
    cache_available = cache_manager.is_cache_available()
    
    if cache_available:
        print("🚀 Using cached embeddings for faster execution")
        
        # Check coverage for your questions
        your_question_ids = ["q1", "q2", "q3"]  # Your actual question IDs
        coverage = cache_manager.get_cache_coverage(
            your_question_ids, config.default_embedding_config
        )
        print(f"Cache coverage: {coverage['coverage_percentage']:.1f}%")
    else:
        print("⚠️  Cache not available, will generate embeddings on-demand")
    
    for question_id in your_question_ids:
        # Try to load cached embeddings first
        embeddings = None
        if cache_available:
            embeddings = cache_manager.get_embeddings_for_question(
                question_id, config.default_embedding_config
            )
        
        if embeddings:
            print(f"✅ Using cached embeddings for {question_id}")
            # Use the cached embeddings in your algorithm
            process_with_cached_embeddings(embeddings)
        else:
            print(f"🔄 Generating embeddings on-demand for {question_id}")
            # Fall back to your original embedding generation
            embeddings = generate_embeddings_yourself(question_id)
            process_with_fresh_embeddings(embeddings)

def process_with_cached_embeddings(embeddings):
    # Your code that uses the embeddings
    pass

def process_with_fresh_embeddings(embeddings):
    # Your code that uses the embeddings
    pass

def generate_embeddings_yourself(question_id):
    # Your original embedding generation code
    pass
'''
    
    print(example_code)
    print()

def main():
    """Run all examples"""
    print("🎯 LongMemEval Embedding Cache System - Usage Examples")
    print("=" * 70)
    print()
    
    try:
        # Run examples
        example_basic_usage()
        example_load_specific_embeddings()
        example_coverage_analysis()
        example_convenience_functions()
        example_integration_pattern()
        
        print("✨ Examples completed!")
        print("=" * 70)
        print("💡 Tips:")
        print("   • Always check cache availability before using")
        print("   • Use coverage analysis to understand cache effectiveness")
        print("   • Fall back to on-demand generation when cache misses occur")
        print("   • Run precompute_embeddings.py to improve cache coverage")
        print()
        print("📚 For more information, see EMBEDDING_CACHE_README.md")
        
    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 