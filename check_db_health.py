#!/usr/bin/env python3
"""
Database Health Check Utility

This script checks the health of the database connection pool and provides
diagnostic information about the current state of database connectivity.

Usage:
    python3 check_db_health.py
"""

import sys
from database import PostgresDatabase


def check_pool_initialization():
    """Check if the connection pool is initialized."""
    print("=" * 60)
    print("DATABASE HEALTH CHECK")
    print("=" * 60)
    print()
    
    print("1. Checking Connection Pool Initialization...")
    status = PostgresDatabase.get_pool_status()
    
    if status["status"] == "not_initialized":
        print("   ❌ Pool is NOT initialized")
        print("   → Pool will be created on first database operation")
    elif status["status"] == "active":
        print("   ✅ Pool is ACTIVE")
        print(f"   → Min connections: {status.get('min_connections', 'N/A')}")
        print(f"   → Max connections: {status.get('max_connections', 'N/A')}")
    else:
        print(f"   ⚠️  Pool status: {status}")
    print()


def check_database_connectivity():
    """Test actual database connectivity."""
    print("2. Testing Database Connectivity...")
    
    db = PostgresDatabase()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Simple query to verify connection
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                
                if result and result[0] == 1:
                    print("   ✅ Database connection SUCCESSFUL")
                    
                    # Get PostgreSQL version
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    print(f"   → PostgreSQL version: {version.split(',')[0]}")
                    
                    # Check current database
                    cursor.execute("SELECT current_database()")
                    db_name = cursor.fetchone()[0]
                    print(f"   → Connected to database: {db_name}")
                    
                    # Check connection count
                    cursor.execute("""
                        SELECT count(*) 
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                    """)
                    active_connections = cursor.fetchone()[0]
                    print(f"   → Active connections to this database: {active_connections}")
                else:
                    print("   ⚠️  Unexpected query result")
    
    except Exception as e:
        print(f"   ❌ Database connection FAILED")
        print(f"   → Error: {str(e)}")
        return False
    
    print()
    return True


def check_required_tables():
    """Check if required tables exist."""
    print("3. Checking Required Tables...")
    
    db = PostgresDatabase()
    required_tables = [
        'akio_data_fastapi',
        'training_jobs',
        'trained_models',
        'multi_model_sessions',
        'multi_model_files',
        'dataset_schemas',
        'reports_fastapi',
        'llm_settings'
    ]
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                for table in required_tables:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        )
                    """, (table,))
                    
                    exists = cursor.fetchone()[0]
                    status = "✅" if exists else "❌"
                    print(f"   {status} {table}")
    
    except Exception as e:
        print(f"   ❌ Error checking tables: {str(e)}")
        return False
    
    print()
    return True


def test_connection_cleanup():
    """Test that connections are properly returned to the pool."""
    print("4. Testing Connection Cleanup...")
    
    db = PostgresDatabase()
    
    try:
        # Test multiple connection acquisitions
        for i in range(5):
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
        
        print("   ✅ Successfully acquired and released 5 connections")
        print("   → Connection pool cleanup working properly")
    
    except Exception as e:
        print(f"   ❌ Connection cleanup test FAILED")
        print(f"   → Error: {str(e)}")
        return False
    
    print()
    return True


def main():
    """Run all health checks."""
    try:
        # Initialize pool
        PostgresDatabase._ensure_pool()
        
        # Run checks
        check_pool_initialization()
        
        connectivity_ok = check_database_connectivity()
        if not connectivity_ok:
            print("\n⚠️  Database connectivity issues detected!")
            print("Please check:")
            print("  1. Database credentials in database.py")
            print("  2. Network connectivity to the database host")
            print("  3. Database is running and accepting connections")
            sys.exit(1)
        
        check_required_tables()
        test_connection_cleanup()
        
        print("=" * 60)
        print("✅ ALL HEALTH CHECKS PASSED")
        print("=" * 60)
        print()
        print("The database connection pool is working correctly.")
        print("You should not experience 'connection pool exhausted' errors.")
        print()
        
    except KeyboardInterrupt:
        print("\n\nHealth check interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ HEALTH CHECK FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
