"""Script to drop all database tables and trigger re-ingestion."""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text

import app.db.session as db_session
from app.db.models import Base
from app.db.session import close_db, get_db, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def drop_all_tables():
    """Drop all tables including dynamic data tables."""
    logger.info("=" * 60)
    logger.info("DROPPING ALL DATABASE TABLES")
    logger.info("=" * 60)

    async with get_db() as session:
        # First, get list of all data tables from registry (if it exists)
        data_tables = []
        try:
            result = await session.execute(text("SELECT data_table FROM data_table_registry"))
            data_tables = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(data_tables)} dynamic data tables to drop")
        except Exception as e:
            logger.info(f"Registry table not found or empty: {e}")

        # Drop all dynamic data tables first
        for table_name in data_tables:
            try:
                await session.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                logger.info(f"  Dropped table: {table_name}")
            except Exception as e:
                logger.warning(f"  Failed to drop {table_name}: {e}")

        await session.commit()

    # Drop all SQLAlchemy-managed tables
    logger.info("Dropping all metadata tables...")
    if db_session.engine:
        async with db_session.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    else:
        raise RuntimeError("Database engine not initialized")

    logger.info("✓ All tables dropped successfully")


async def recreate_tables():
    """Recreate all SQLAlchemy-managed tables."""
    logger.info("=" * 60)
    logger.info("RECREATING METADATA TABLES")
    logger.info("=" * 60)

    if db_session.engine:
        async with db_session.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:
        raise RuntimeError("Database engine not initialized")

    logger.info("✓ Metadata tables recreated successfully")


async def trigger_ingestion():
    """Trigger data ingestion."""
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION")
    logger.info("=" * 60)

    from app.services.ingestion.data_processor import DataProcessor

    async with get_db() as session:
        processor = DataProcessor()
        counts = await processor.process_all_datasets(session)

        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Metadata entries: {counts['metadata_entries']}")
        logger.info(f"Data tables: {counts['data_tables']}")
        logger.info(f"Total rows: {counts['total_rows']:,}")


async def create_indexes():
    """Create vector and full-text search indexes."""
    logger.info("=" * 60)
    logger.info("CREATING SEARCH INDEXES")
    logger.info("=" * 60)

    async with get_db() as session:
        categories = ["employment", "income", "hours_worked"]

        for category in categories:
            table_name = f"{category}_dataset_metadata"

            try:
                # Create IVFFlat vector index for similarity search
                logger.info(f"Creating vector index on {table_name}...")
                await session.execute(
                    text(
                        f"""
                    CREATE INDEX IF NOT EXISTS {category}_embedding_idx
                    ON {table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10)
                """
                    )
                )

                # Create GIN index for full-text search
                logger.info(f"Creating full-text search index on {table_name}...")
                await session.execute(
                    text(
                        f"""
                    CREATE INDEX IF NOT EXISTS {category}_tsv_idx
                    ON {table_name}
                    USING gin(tsv)
                """
                    )
                )

                await session.commit()
                logger.info(f"✓ Indexes created for {table_name}")

            except Exception as e:
                logger.warning(f"Failed to create indexes for {table_name}: {e}")
                await session.rollback()


async def main():
    """Main execution."""
    logger.info("\n" + "=" * 60)
    logger.info("DATABASE RESET AND RE-INGESTION")
    logger.info("=" * 60 + "\n")

    # Initialize database connection
    init_db()

    try:
        # Step 1: Drop all tables
        await drop_all_tables()

        # Step 2: Recreate metadata tables
        await recreate_tables()

        # Step 3: Trigger ingestion
        await trigger_ingestion()

        # Step 4: Create indexes
        await create_indexes()

        logger.info("\n" + "=" * 60)
        logger.info("✓ DATABASE RESET COMPLETE")
        logger.info("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"\n✗ Database reset failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
