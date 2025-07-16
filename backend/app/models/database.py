"""
Database configuration and connection management
"""
import asyncpg
import os
from typing import Optional

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = os.getenv("DATABASE_URL", 
                                     "postgresql://localhost/gct_saas")
    
    async def connect(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.connect()
        return self.pool.acquire()

# Global database manager instance
db_manager = DatabaseManager()
