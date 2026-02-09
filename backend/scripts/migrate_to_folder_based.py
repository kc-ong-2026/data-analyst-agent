"""Migration script to convert from 3-tier to folder-based architecture.

This script:
1. Drops old tables (dataset_metadata, data_chunks, raw_data_tables, raw_*)
2. Executes new schema from init_folder_based.sql
3. Triggers re-ingestion of all datasets

CAUTION: This script drops all existing data and re-ingests from CSV/Excel files.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import config
from app.services.ingestion.data_processor import DataProcessor
from app.services.ingestion.embedding_generator import EmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def drop_old_schema(session: AsyncSession):
    """Drop all old tables from 3-tier architecture."""
    logger.info("Dropping old schema...")

    # Get list of all raw_* tables
    result = await session.execute(
        text("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public' AND tablename LIKE 'raw_%'
        """)
    )
    raw_tables = [row[0] for row in result.fetchall()]

    # Drop all raw_* tables
    for table_name in raw_tables:
        logger.info(f"Dropping table: {table_name}")
        await session.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))

    # Drop old core tables
    old_tables = ["raw_data_tables", "data_chunks", "dataset_metadata"]
    for table_name in old_tables:
        logger.info(f"Dropping table: {table_name}")
        await session.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))

    await session.commit()
    logger.info("Old schema dropped successfully")


async def create_new_schema(session: AsyncSession):
    """Execute new schema from init_folder_based.sql."""
    logger.info("Creating new folder-based schema...")

    # Read SQL file
    sql_file = backend_dir / "db" / "init_folder_based.sql"
    if not sql_file.exists():
        raise FileNotFoundError(f"Schema file not found: {sql_file}")

    sql_content = sql_file.read_text()

    # Split by semicolons and execute each statement
    statements = [s.strip() for s in sql_content.split(";") if s.strip()]

    for statement in statements:
        # Skip comments and empty statements
        if not statement or statement.startswith("--"):
            continue

        try:
            await session.execute(text(statement))
        except Exception as e:
            logger.error(f"Failed to execute statement: {statement[:100]}...")
            logger.error(f"Error: {e}")
            raise

    await session.commit()
    logger.info("New schema created successfully")


async def ingest_datasets(session: AsyncSession):
    """Re-ingest all datasets using the new folder-based processor."""
    logger.info("Starting dataset ingestion...")

    processor = DataProcessor()
    counts = await processor.process_all_datasets(session)

    await session.commit()

    logger.info(f"Ingestion complete: {counts}")
    return counts


async def generate_embeddings(session: AsyncSession):
    """Generate embeddings for all folder metadata."""
    logger.info("Generating embeddings...")

    generator = EmbeddingGenerator()
    counts = await generator.update_embeddings(session)

    await session.commit()

    logger.info(f"Embeddings generated: {counts}")
    return counts


async def verify_migration(session: AsyncSession):
    """Verify the migration was successful."""
    logger.info("Verifying migration...")

    # Check folder metadata tables
    folder_tables = [
        "employment_dataset_metadata",
        "hours_worked_dataset_metadata",
        "income_dataset_metadata",
        "labour_force_dataset_metadata",
    ]

    for table_name in folder_tables:
        result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()
        logger.info(f"{table_name}: {count} rows")

    # Check registry table
    result = await session.execute(text("SELECT COUNT(*) FROM data_table_registry"))
    registry_count = result.scalar()
    logger.info(f"data_table_registry: {registry_count} rows")

    # Check for data tables
    result = await session.execute(
        text("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND (tablename LIKE 'employment_%'
                 OR tablename LIKE 'hours_worked_%'
                 OR tablename LIKE 'income_%'
                 OR tablename LIKE 'labour_force_%')
            AND tablename NOT LIKE '%_metadata'
        """)
    )
    data_tables = [row[0] for row in result.fetchall()]
    logger.info(f"Found {len(data_tables)} data tables")

    if registry_count == 0:
        logger.warning("No entries in data_table_registry!")
    elif len(data_tables) == 0:
        logger.warning("No data tables created!")
    else:
        logger.info("✓ Migration verification passed")


async def main():
    """Run the migration."""
    logger.info("=" * 60)
    logger.info("Starting migration to folder-based architecture")
    logger.info("=" * 60)

    # Create database engine
    database_url = config.database_url
    if not database_url:
        logger.error("DATABASE_URL not configured")
        return

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    try:
        async with async_session() as session:
            # Step 1: Drop old schema
            await drop_old_schema(session)

            # Step 2: Create new schema
            await create_new_schema(session)

            # Step 3: Ingest datasets
            counts = await ingest_datasets(session)

            # Step 4: Generate embeddings
            embedding_counts = await generate_embeddings(session)

            # Step 5: Verify migration
            await verify_migration(session)

        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info(f"Ingested {counts.get('metadata_entries', 0)} datasets")
        logger.info(f"Created {counts.get('data_tables', 0)} data tables")
        logger.info(f"Total rows: {counts.get('total_rows', 0):,}")
        logger.info(f"Generated {embedding_counts.get('total', 0)} embeddings")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
