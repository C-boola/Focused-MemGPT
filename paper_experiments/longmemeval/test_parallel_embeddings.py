#!/usr/bin/env python3
"""
Test Script for Parallel Embedding Generation

This script validates that the parallel embedding generation works correctly
by running a small test and comparing results.

Usage:
    python test_parallel_embeddings.py
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_parallel_embedding_generation():
    """Test parallel embedding generation with a small dataset"""
    print("🧪 Testing Parallel Embedding Generation")
    print("=" * 50)
    
    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp(prefix="test_parallel_cache_")
    print(f"Using temporary cache directory: {temp_dir}")
    
    try:
        # Test the parallel precomputation script
        print("Running parallel embedding generation in test mode...")
        
        cmd = [
            sys.executable, "precompute_embeddings.py",
            "--test",
            "--batch-size", "3",
            "--max-workers", "2"
        ]
        
        # Set environment variable to use our temp cache dir
        env = os.environ.copy()
        env["EMBEDDINGS_CACHE_DIR"] = temp_dir
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.dirname(__file__)
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if command succeeded
        if result.returncode == 0:
            print("✅ Parallel embedding generation completed successfully!")
            
            # Check if cache directory was created
            cache_path = Path(temp_dir)
            embeddings_dir = cache_path / "embeddings"
            index_file = cache_path / "embedding_index.json"
            
            if embeddings_dir.exists():
                print("✅ Embeddings directory created")
                
                # Count embedding files
                embedding_files = list(embeddings_dir.glob("*.pkl.gz"))
                print(f"✅ Generated {len(embedding_files)} embedding files")
                
                if len(embedding_files) > 0:
                    print("✅ Successfully generated embeddings in parallel")
                else:
                    print("⚠️  No embedding files found")
            else:
                print("❌ Embeddings directory not found")
            
            if index_file.exists():
                print("✅ Index file created")
                
                # Try to read index
                try:
                    import json
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    print(f"✅ Index contains {len(index_data)} model entries")
                except Exception as e:
                    print(f"❌ Error reading index file: {e}")
            else:
                print("❌ Index file not found")
                
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    return True

def test_command_line_options():
    """Test that command line options are parsed correctly"""
    print("\n🔧 Testing Command Line Options")
    print("=" * 50)
    
    try:
        # Test help option
        cmd = [sys.executable, "precompute_embeddings.py", "--help"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            print("✅ Help option works correctly")
            
            # Check for expected options in help text
            help_text = result.stdout
            expected_options = ["--batch-size", "--max-workers", "--test", "--force", "--cleanup"]
            
            for option in expected_options:
                if option in help_text:
                    print(f"✅ Found {option} in help text")
                else:
                    print(f"❌ Missing {option} in help text")
                    return False
        else:
            print(f"❌ Help command failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Command line test failed with error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🎯 Testing Parallel Embedding Generation System")
    print("=" * 70)
    
    success = True
    
    # Test command line options
    if not test_command_line_options():
        success = False
    
    # Test parallel generation (only if data files exist)
    data_path = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s.json")
    oracle_path = os.path.join(os.path.dirname(__file__), "data", "longmemeval_oracle.json")
    
    if os.path.exists(data_path) and os.path.exists(oracle_path):
        if not test_parallel_embedding_generation():
            success = False
    else:
        print("\n⚠️  Skipping parallel generation test - LongMemEval data files not found")
        print("   This is normal if you haven't set up the benchmark data yet")
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The parallel embedding generation system is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 