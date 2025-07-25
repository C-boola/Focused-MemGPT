# Database Performance Optimizations

This document outlines the performance optimizations implemented in the MemGPT database connectors to increase concurrent connections and improve overall performance.

## Overview

The optimizations focus on:
1. **Connection Pooling**: Improved connection management for concurrent access
2. **Query Performance**: Database indexes and optimized queries
3. **Bulk Operations**: Efficient batch processing for large datasets
4. **Session Management**: Thread-safe session handling
5. **Database-Specific Tuning**: PostgreSQL and SQLite specific optimizations

## Connection Pooling Improvements

### PostgreSQL Connector
- **Pool Size**: Configurable via `MEMGPT_PG_POOL_SIZE` (default: 20 connections)
- **Max Overflow**: Additional connections via `MEMGPT_PG_MAX_OVERFLOW` (default: 30)
- **Pool Timeout**: Connection timeout via `MEMGPT_PG_POOL_TIMEOUT` (default: 30s)
- **Pool Recycle**: Connection recycling every hour to prevent stale connections
- **Pre-ping**: Validates connections before use
- **Connection Reset**: Automatic cleanup on connection return

### SQLite Connector
- **Static Pool**: Single connection pool suitable for SQLite's architecture
- **WAL Mode**: Write-Ahead Logging for better concurrency
- **Memory Mapping**: 256MB memory map for improved I/O performance
- **Cache Size**: 10MB cache for better query performance

## Environment Variables for Configuration

```bash
# PostgreSQL Connection Pool Settings
export MEMGPT_PG_POOL_SIZE=20        # Base number of connections
export MEMGPT_PG_MAX_OVERFLOW=30     # Additional connections beyond pool_size
export MEMGPT_PG_POOL_TIMEOUT=30     # Timeout for getting connection (seconds)
export MEMGPT_PG_POOL_RECYCLE=3600   # Recycle connections after 1 hour
```

## Database Indexes

### Automatic Index Creation
- **Single Column Indexes**: Added on frequently queried columns
  - `user_id`, `agent_id`, `created_at`, `role`, `tool_call_id`, `data_source`, `doc_id`
  
- **Composite Indexes**: Multi-column indexes for common query patterns
  - `(user_id, agent_id)`: For user-agent specific queries
  - `(user_id, created_at DESC)`: For user timeline queries
  - `(agent_id, created_at DESC)`: For agent timeline queries
  - `(role, created_at DESC)`: For role-based message queries

### Benefits
- Faster WHERE clause evaluation
- Improved ORDER BY performance
- Reduced table scan operations
- Better query plan optimization

## Bulk Operations

### PostgreSQL Optimizations
- **COPY Command**: For datasets > 1000 records, uses PostgreSQL's COPY command
- **Chunked Processing**: Large datasets processed in manageable chunks
- **Upsert Operations**: Efficient ON CONFLICT handling for updates
- **Bulk Mappings**: SQLAlchemy's bulk_insert_mappings for smaller batches

### SQLite Optimizations
- **Transaction Batching**: Groups operations in single transactions
- **Chunk Processing**: 500-record chunks to balance memory and performance
- **Pragma Optimizations**: Database-level performance tuning

## Session Management

### Thread-Safe Sessions
- **Scoped Sessions**: Thread-local session management
- **Context Managers**: Automatic transaction handling with rollback on errors
- **Connection Reuse**: Efficient connection pooling and reuse
- **Manual Flush Control**: Better control over when data is written

### Session Configuration
- `expire_on_commit=False`: Prevents object expiration for better performance
- `autoflush=False`: Manual control over flush operations
- Thread-local storage for session management

## Query Optimizations

### Improved Query Patterns
- **Count Optimization**: Uses `func.count()` instead of `.count()` for better performance
- **Bulk Delete**: `synchronize_session=False` for faster delete operations
- **Vector Similarity**: Optimized L2 distance queries for embeddings
- **Pagination**: Efficient offset/limit patterns

### Database-Specific Optimizations

#### PostgreSQL
- **JIT Disabled**: Better performance for short queries
- **Application Name**: Connection identification for monitoring
- **Connection Timeouts**: Proper timeout handling

#### SQLite
- **WAL Mode**: Write-Ahead Logging for better concurrency
- **Synchronous Mode**: NORMAL for balance of safety and performance
- **Memory Store**: Temporary tables stored in memory
- **Optimize Pragma**: Regular database optimization

## Performance Monitoring

### Connection Pool Monitoring
The connection pools can be monitored through SQLAlchemy's built-in metrics:
- Pool size and overflow
- Connection checkout time
- Active connections

### Database-Specific Monitoring
- **PostgreSQL**: Use `pg_stat_activity` to monitor connections
- **SQLite**: Monitor file locks and WAL file size

## Usage Examples

### Basic Usage (No Changes Required)
```python
# Existing code continues to work without modification
connector = PostgresStorageConnector("recall_memory", config, user_id, agent_id)
records = connector.get_all(filters={"agent_id": agent_id})
```

### Bulk Operations
```python
# Optimized bulk inserts automatically used
large_dataset = [Message(...) for _ in range(5000)]
connector.insert_many(large_dataset)  # Uses optimized bulk insert
```

### Environment Configuration
```python
# Configure connection pool size via environment
os.environ["MEMGPT_PG_POOL_SIZE"] = "50"  # Increase for high concurrency
os.environ["MEMGPT_PG_MAX_OVERFLOW"] = "100"
```

## Performance Impact

### Expected Improvements
- **Concurrent Connections**: 10-50x increase in supported concurrent connections
- **Bulk Insert Speed**: 5-20x faster for large datasets
- **Query Performance**: 2-5x improvement on indexed columns
- **Memory Usage**: Reduced memory footprint through better session management

### Backward Compatibility
All optimizations are backward compatible. Existing code will automatically benefit from performance improvements without any code changes.

## Troubleshooting

### Connection Pool Issues
- Monitor pool exhaustion via SQLAlchemy logging
- Adjust `MEMGPT_PG_POOL_SIZE` and `MEMGPT_PG_MAX_OVERFLOW` based on load
- Check for connection leaks in application code

### SQLite Lock Issues
- Ensure WAL mode is enabled
- Check for long-running transactions
- Monitor SQLite file permissions

### Performance Degradation
- Run `PRAGMA optimize` regularly for SQLite
- Monitor index usage in PostgreSQL
- Check for missing composite indexes on new query patterns 