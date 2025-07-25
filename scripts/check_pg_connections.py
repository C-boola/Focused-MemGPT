#!/usr/bin/env python3
"""
Utility script to check PostgreSQL connection limits and diagnose connection pool issues.
Run this to help debug "too many clients" errors.
"""

import os
import sys
import uuid
from pathlib import Path

# Add the parent directory to Python path to import memgpt modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from memgpt.config import MemGPTConfig
from memgpt.agent_store.db import PostgresStorageConnector
from memgpt.agent_store.storage import TableType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pg_max_connections(connector: PostgresStorageConnector):
    """Check PostgreSQL max_connections setting"""
    try:
        with connector.session_maker() as session:
            from sqlalchemy import text
            
            result = session.execute(text("SHOW max_connections;")).fetchone()
            max_connections = int(result[0])
            
            # Get current connection count
            result = session.execute(text("""
                SELECT count(*) 
                FROM pg_stat_activity 
                WHERE state = 'active' OR state = 'idle in transaction'
            """)).fetchone()
            current_connections = int(result[0])
            
            # Get connections by application name
            result = session.execute(text("""
                SELECT application_name, count(*) 
                FROM pg_stat_activity 
                WHERE application_name LIKE 'memgpt%'
                GROUP BY application_name
                ORDER BY count(*) DESC
            """)).fetchall()
            
            print(f"PostgreSQL Connection Status:")
            print(f"  Max connections: {max_connections}")
            print(f"  Current connections: {current_connections}")
            print(f"  Percentage used: {(current_connections/max_connections)*100:.1f}%")
            print(f"  Available connections: {max_connections - current_connections}")
            print()
            
            if result:
                print("MemGPT connections by type:")
                for app_name, count in result:
                    print(f"  {app_name}: {count}")
            else:
                print("No MemGPT connections found")
            print()
            
            return max_connections, current_connections
            
    except Exception as e:
        logger.error(f"Failed to check PostgreSQL connections: {e}")
        return None, None

def test_connection_pools():
    """Test connection pool configuration"""
    try:
        config = MemGPTConfig.load()
        user_id = uuid.uuid4()
        
        print("Testing connection pools...")
        
        # Test different table types
        connectors = []
        try:
            # Test recall memory connector
            recall_connector = PostgresStorageConnector(
                table_type=TableType.RECALL_MEMORY, 
                config=config, 
                user_id=user_id, 
                agent_id=uuid.uuid4()
            )
            connectors.append(("Recall Memory", recall_connector))
            
            # Test archival memory connector  
            archival_connector = PostgresStorageConnector(
                table_type=TableType.ARCHIVAL_MEMORY,
                config=config,
                user_id=user_id,
                agent_id=uuid.uuid4()  # Archival memory requires agent_id
            )
            connectors.append(("Archival Memory", archival_connector))
            
            print("\nConnection Pool Status:")
            for name, connector in connectors:
                pool_status = connector.get_pool_status()
                print(f"  {name}:")
                if 'error' in pool_status:
                    print(f"    Status: {pool_status['error']}")
                    print(f"    Pool class: {pool_status.get('pool_class', 'Unknown')}")
                else:
                    print(f"    Pool size: {pool_status['size']}")
                    print(f"    Checked in: {pool_status['checked_in']}")
                    print(f"    Checked out: {pool_status['checked_out']}")
                    print(f"    Total active: {pool_status['total']}")
                    print(f"    Overflow: {pool_status.get('overflow', 'N/A')}")
                    print(f"    Max overflow: {pool_status.get('max_overflow', 'N/A')}")
            
            # Check PostgreSQL max connections
            max_conn, current_conn = check_pg_max_connections(connectors[0][1])
            
            if max_conn and current_conn:
                total_possible_connections = sum(
                    int(os.getenv("MEMGPT_PG_POOL_SIZE", "5")) + 
                    int(os.getenv("MEMGPT_PG_MAX_OVERFLOW", "10"))
                    for _ in connectors
                )
                
                print(f"\nConnection Analysis:")
                print(f"  Max connections per pool: {int(os.getenv('MEMGPT_PG_POOL_SIZE', '5')) + int(os.getenv('MEMGPT_PG_MAX_OVERFLOW', '10'))}")
                print(f"  Number of pools: {len(connectors)}")
                print(f"  Max possible MemGPT connections: {total_possible_connections}")
                print(f"  PostgreSQL max_connections: {max_conn}")
                
                if total_possible_connections > max_conn * 0.8:  # If we could use >80% of connections
                    print("  ⚠️  WARNING: Pool configuration may exceed PostgreSQL limits!")
                    print("     Consider reducing MEMGPT_PG_POOL_SIZE and MEMGPT_PG_MAX_OVERFLOW")
                else:
                    print("  ✅ Pool configuration looks reasonable")
                    
        finally:
            # Clean up test tables
            for name, connector in connectors:
                try:
                    connector.delete_table()
                    logger.info(f"Cleaned up test table for {name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up test table for {name}: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to test connection pools: {e}")

def main():
    """Main function"""
    print("PostgreSQL Connection Diagnostics")
    print("=" * 40)
    
    # Check environment variables
    print("Environment Configuration:")
    print(f"  MEMGPT_PG_POOL_SIZE: {os.getenv('MEMGPT_PG_POOL_SIZE', '5 (default)')}")
    print(f"  MEMGPT_PG_MAX_OVERFLOW: {os.getenv('MEMGPT_PG_MAX_OVERFLOW', '10 (default)')}")
    print(f"  MEMGPT_PG_POOL_TIMEOUT: {os.getenv('MEMGPT_PG_POOL_TIMEOUT', '30 (default)')}")
    print(f"  MEMGPT_PG_POOL_RECYCLE: {os.getenv('MEMGPT_PG_POOL_RECYCLE', '1800 (default)')}")
    print()
    
    test_connection_pools()
    
    print("\nRecommendations:")
    print("1. If you see 'too many clients' errors:")
    print("   - Reduce MEMGPT_PG_POOL_SIZE (current: {})".format(os.getenv('MEMGPT_PG_POOL_SIZE', '5')))
    print("   - Reduce MEMGPT_PG_MAX_OVERFLOW (current: {})".format(os.getenv('MEMGPT_PG_MAX_OVERFLOW', '10')))
    print("   - Increase PostgreSQL max_connections setting")
    print("2. For debugging, enable connection logging:")
    print("   - Set MEMGPT_PG_ECHO=true")
    print("   - Set MEMGPT_PG_ECHO_POOL=true")
    print("3. Monitor connection usage with:")
    print("   - SELECT * FROM pg_stat_activity WHERE application_name LIKE 'memgpt%';")

if __name__ == "__main__":
    main() 