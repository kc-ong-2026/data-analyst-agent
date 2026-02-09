"""SQLAlchemy ORM models for the RAG system."""

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship

VECTOR_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(Text, nullable=False, unique=True)
    file_path = Column(Text, nullable=False)
    category = Column(Text)
    description = Column(Text)
    columns = Column(JSONB, default=list)
    row_count = Column(Integer, default=0)
    year_range = Column(JSONB, default=dict)
    dimensions = Column(JSONB, default=dict)
    summary_text = Column(Text)
    embedding = Column(Vector(VECTOR_DIMENSIONS))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chunks = relationship("DataChunk", back_populates="dataset", cascade="all, delete-orphan")
    raw_tables = relationship("RawDataTable", back_populates="dataset", cascade="all, delete-orphan")


class DataChunk(Base):
    __tablename__ = "data_chunks"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("dataset_metadata.id", ondelete="CASCADE"), nullable=False)
    chunk_type = Column(Text, nullable=False, default="group")
    group_key = Column(JSONB, default=dict)
    chunk_text = Column(Text, nullable=False)
    row_count = Column(Integer, default=0)
    numeric_summary = Column(JSONB, default=dict)
    embedding = Column(Vector(VECTOR_DIMENSIONS))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    dataset = relationship("DatasetMetadata", back_populates="chunks")


class RawDataTable(Base):
    __tablename__ = "raw_data_tables"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("dataset_metadata.id", ondelete="CASCADE"), nullable=False)
    table_name = Column(Text, nullable=False, unique=True)
    columns = Column(JSONB, default=list)
    row_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    dataset = relationship("DatasetMetadata", back_populates="raw_tables")
