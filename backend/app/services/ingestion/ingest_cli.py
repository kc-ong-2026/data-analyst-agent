"""CLI entry point for data ingestion."""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_ingestion() -> None:
    """Run the full ingestion pipeline."""
    from app.db.session import close_db, get_db, init_db
    from app.services.ingestion.data_processor import DataProcessor
    from app.services.ingestion.embedding_generator import EmbeddingGenerator

    logger.info("Starting data ingestion pipeline...")

    init_db()

    try:
        async with get_db() as session:
            # Step 1: Process datasets
            processor = DataProcessor()
            counts = await processor.process_all_datasets(session)
            logger.info(
                f"Processed {counts['metadata_entries']} metadata entries, "
                f"{counts['data_tables']} data tables, "
                f"{counts['total_rows']} total rows"
            )

            # Step 2: Generate embeddings
            generator = EmbeddingGenerator()
            embed_counts = await generator.update_embeddings(session)
            logger.info(
                f"Generated {embed_counts.get('total', 0)} embeddings across all categories"
            )

        logger.info("Ingestion complete!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(run_ingestion())
