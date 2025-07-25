# LongMemEval Embedding Cache System

This system provides efficient caching and reuse of vector embeddings for the LongMemEval benchmark, significantly reducing execution time and API costs when running embedding-dependent memory modes (focus, hybrid, density, pure_cluster).

## Overview

The embedding cache system consists of several components:

1. **`precompute_embeddings.py`** - Precomputes and stores embeddings for all test cases
2. **`embedding_utils.py`** - Utilities for accessing cached embeddings  
3. **`run_longmemeval_cached.py`** - Enhanced benchmark script that uses cached embeddings
4. **`agent_cache_patch.py`** - Patches for the Agent class to support cached embeddings

## Quick Start

### 1. Precompute Embeddings

First, generate embeddings for all test cases using parallel processing:

```bash
# Generate embeddings for all test cases (default: 10 parallel workers)
python precompute_embeddings.py

# Generate embeddings with custom batch size and workers
python precompute_embeddings.py --batch-size 20 --max-workers 8

# Generate embeddings for test subset only (faster)
python precompute_embeddings.py --test

# Force regeneration of existing embeddings
python precompute_embeddings.py --force

# Clean up orphaned cache files
python precompute_embeddings.py --cleanup

# Combine options for fast parallel processing
python precompute_embeddings.py --test --batch-size 15 --max-workers 6
```

### 2. Run Benchmark with Cached Embeddings

Use the cache-optimized benchmark script:

```bash
# Run with focus mode (uses cached embeddings)
python run_longmemeval_cached.py --mode focus

# Run with hybrid mode (uses cached embeddings when beta > 0)
python run_longmemeval_cached.py --mode hybrid --beta 0.7

# Run with FIFO mode (no embeddings needed)
python run_longmemeval_cached.py --mode fifo

# Test run with cached embeddings
python run_longmemeval_cached.py --test --mode focus
```

## Benefits

- **Speed**: Avoid regenerating embeddings for each run (5-10x faster for embedding-dependent modes)
- **Cost**: Eliminate redundant API calls to embedding services
- **Consistency**: Use identical embeddings across different runs for fair comparisons
- **Flexibility**: Fall back to on-demand generation when cache misses occur

## Cache Structure

```
embeddings_cache/
├── embeddings/                    # Compressed embedding files
│   ├── question_001_abc12345.pkl.gz
│   ├── question_002_abc12345.pkl.gz
│   └── ...
├── metadata/                      # Additional metadata (future use)
├── embedding_index.json           # Index mapping questions to files
└── cache_config.json             # Cache configuration
```

### Cache Files

Each cached embedding file contains:
- Question ID and metadata
- List of (user_msg_id, assistant_msg_id, embedding_vector) tuples
- Embedding model configuration
- Generation timestamp

## Memory Modes and Embedding Usage

| Memory Mode | Requires Embeddings | Cache Benefit |
|-------------|-------------------|---------------|
| `fifo` | ❌ No | None |
| `focus` | ✅ Yes | High |
| `hybrid` (β=0) | ❌ No | None |
| `hybrid` (β>0) | ✅ Yes | High |
| `pure_cluster` | ✅ Yes | High |
| `density` (β=0) | ❌ No | None |
| `density` (β>0) | ✅ Yes | High |

## API Reference

### EmbeddingCacheManager

```python
from embedding_utils import EmbeddingCacheManager

cache_manager = EmbeddingCacheManager()

# Check if cache is available
if cache_manager.is_cache_available():
    print("Cache is ready!")

# Get embeddings for a question
embeddings = cache_manager.get_embeddings_for_question(
    question_id="question_001",
    embedding_config=config.default_embedding_config
)

# Check cache coverage
coverage = cache_manager.get_cache_coverage(
    all_question_ids=["q1", "q2", "q3"],
    embedding_config=config.default_embedding_config
)
print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
```

### Convenience Functions

```python
from embedding_utils import load_cached_embeddings, check_embedding_cache_coverage

# Simple loading
embeddings = load_cached_embeddings("question_001", embedding_config)

# Check coverage
coverage = check_embedding_cache_coverage(all_question_ids, embedding_config)
```

## Integration Examples

### Using in Custom Scripts

```python
from embedding_utils import EmbeddingCacheManager
from memgpt.config import MemGPTConfig

# Initialize
config = MemGPTConfig.load()
cache_manager = EmbeddingCacheManager()

# Check what's available
if cache_manager.is_cache_available():
    cache_info = cache_manager.get_cache_info()
    print(f"Found {len(cache_info['models'])} embedding models in cache")

# Load embeddings for specific question
question_id = "question_001"
embeddings = cache_manager.get_embeddings_for_question(
    question_id, config.default_embedding_config
)

if embeddings:
    print(f"Loaded {len(embeddings)} embedding pairs")
    for user_id, assistant_id, vector in embeddings:
        print(f"User {user_id} + Assistant {assistant_id} -> {len(vector)}D vector")
```

### Agent Patching (Advanced)

```python
from agent_cache_patch import patch_agent_with_cache
from memgpt.agent import Agent

# Create agent
agent = Agent(...)

# Patch with cache support
patch_agent_with_cache(agent, cache_manager)

# Agent will now automatically use cached embeddings when available
```

## Cache Management

### Checking Cache Status

```python
from embedding_utils import EmbeddingCacheManager

cache = EmbeddingCacheManager()
info = cache.get_cache_info()

print("Cache Status:")
for model_hash, model_info in info["models"].items():
    print(f"  {model_info['model']}: {model_info['num_questions']} questions")
```

### Cache Statistics

```bash
# View cache contents
python -c "
from embedding_utils import EmbeddingCacheManager
cache = EmbeddingCacheManager()
info = cache.get_cache_info()
print('Cache Info:', info)
"
```

### Cleaning Cache

```bash
# Remove orphaned files
python precompute_embeddings.py --cleanup

# Manually remove entire cache
rm -rf embeddings_cache/
```

## Performance Comparison

### Without Cache (Cold Run)
```
Time per test case: ~15-30 seconds
API calls: ~50-200 embedding requests per case
Total cost: $X.XX per full benchmark run
```

### With Cache (Warm Run)
```
Time per test case: ~3-5 seconds  
API calls: 0 embedding requests
Total cost: Only LLM inference costs
Speedup: 5-10x faster
```

### Cache Generation Performance

**Sequential Processing (Original):**
```
Time to generate full cache: ~8-12 hours
Processing: 1 case at a time
Resource usage: Single core, low memory
```

**Parallel Processing (New):**
```
Time to generate full cache: ~2-4 hours (with 10 workers)
Processing: 10 cases simultaneously by default
Resource usage: Multi-core, higher memory
Speedup: 3-4x faster cache generation
```

## Troubleshooting

### Cache Not Found
```
❌ Embedding cache not available. Embeddings will be generated on-demand.
   Run 'python precompute_embeddings.py' to build the cache.
```
**Solution**: Run the precomputation script to generate the cache.

### Cache Mismatch
```
[CACHE] Warning: Cached embeddings don't match current message sequence, generating fresh embeddings
```
**Solution**: This is normal when message sequences differ. The system will fall back to on-demand generation.

### Partial Coverage
```
⚡ Selected mode will use cached embeddings for 145 cases
   5 cases will generate embeddings on-demand
```
**Solution**: This is normal. You can rerun precomputation with `--force` to regenerate missing embeddings.

### Storage Space
The cache typically uses:
- ~50-100MB per 1000 questions
- ~200-500MB for full LongMemEval dataset
- Files are compressed with gzip

## Configuration

### Parallel Processing Options

The precomputation script supports several options for optimizing performance:

```bash
python precompute_embeddings.py --help
```

**Available Options:**
- `--batch-size N`: Number of test cases to process in parallel (default: 10)
- `--max-workers N`: Maximum number of worker processes (default: same as batch-size)
- `--test`: Run in test mode with limited cases for quick testing
- `--force`: Force regeneration of existing embeddings
- `--cleanup`: Clean up orphaned cache files

**Performance Tuning:**
- **High-end machines**: Use `--batch-size 20 --max-workers 16` for maximum speed
- **Memory-constrained**: Use `--batch-size 5 --max-workers 4` to reduce memory usage
- **Testing**: Use `--test --batch-size 5` for quick validation

### Embedding Model Changes
If you change embedding models, the cache will automatically handle different models separately by using embedding configuration hashes.

### Cache Location
Default: `./embeddings_cache/`

To use a custom location:
```python
cache_manager = EmbeddingCacheManager("/path/to/custom/cache")
```

## Best Practices

1. **Always precompute before long benchmark runs** to maximize performance benefits
2. **Use test mode first** (`--test`) to verify cache functionality and optimal batch size
3. **Monitor cache coverage** to ensure you're getting the expected performance benefits
4. **Clean up periodically** to remove orphaned files (`--cleanup`)
5. **Back up cache** before major changes to embedding configurations
6. **Tune parallel processing** based on your hardware:
   - **CPU cores**: Set `--max-workers` to 50-75% of available cores
   - **Memory**: Reduce `--batch-size` if you encounter memory issues
   - **API limits**: Lower batch size if hitting rate limits
7. **Use appropriate batch sizes**:
   - **Development/testing**: `--batch-size 5`
   - **Standard machines**: `--batch-size 10` (default)
   - **High-performance**: `--batch-size 20+`

## Future Enhancements

- Distributed cache sharing across multiple machines
- Automatic cache warming based on planned experiments
- Cache compression improvements
- Integration with cloud storage for team sharing
- Cache versioning for reproducibility

---

For questions or issues, please refer to the source code documentation or create an issue in the repository. 