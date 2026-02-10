"""Data Extraction Agent - Extracts data using LangGraph workflow with RAG."""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.models import (
    DatasetMetadata,
    ExtractedDataset,
    ExtractionResult,
    YearRange,
)
from ..base_agent import (
    AgentRole,
    AgentResponse,
    AgentState,
    BaseAgent,
    GraphState,
)
from app.services.data_service import data_service
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DataExtractionAgent(BaseAgent):
    """Agent responsible for extracting data from government data sources.

    Uses LangGraph to construct a flow with nodes:
    1. retrieve_context - Semantic + full-text search via RAG (falls back to file matching)
    2. load_raw_data - Load raw data from RAG results or files
    3. extract_relevant_data - Extract specific data based on query
    4. format_output - Format data for analytics agent
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.EXTRACTION

    @property
    def name(self) -> str:
        return "Data Extraction"

    @property
    def description(self) -> str:
        return "Extracts data from Singapore government datasets and APIs"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for data extraction.

        Flow:
        retrieve_context -> load_dataframes -> extract_relevant_data -> format_output -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("load_dataframes", self._load_dataframes_node)
        workflow.add_node("extract_relevant_data", self._extract_relevant_data_node)
        workflow.add_node("format_output", self._format_output_node)

        # Set entry point
        workflow.set_entry_point("retrieve_context")

        # Add edges with conditional routing
        workflow.add_edge("retrieve_context", "load_dataframes")
        workflow.add_conditional_edges(
            "load_dataframes",
            self._should_extract,
            {
                "extract": "extract_relevant_data",
                "skip": "format_output",
            }
        )
        workflow.add_edge("extract_relevant_data", "format_output")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def _should_extract(self, state: GraphState) -> str:
        """Determine if we should extract data or skip to formatting."""
        loaded_data = state.get("intermediate_results", {}).get("loaded_datasets", {})
        logger.info(f"SHOULD_EXTRACT: loaded_datasets keys={list(loaded_data.keys())}, count={len(loaded_data)}")
        if loaded_data:
            logger.info("SHOULD_EXTRACT: routing to 'extract'")
            return "extract"
        logger.info("SHOULD_EXTRACT: routing to 'skip'")
        return "skip"

    async def _retrieve_context_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Retrieve relevant context using RAG (with file-based fallback)."""
        current_task = state.get("current_task", "")
        required_data = state.get("metadata", {}).get("required_data", [])
        workflow_plan = state.get("workflow_plan", [])

        step_task = ""
        current_step = state.get("current_step", 0)
        if workflow_plan and current_step < len(workflow_plan):
            step_task = workflow_plan[current_step].get("task", "")

        search_query = current_task
        if step_task:
            search_query = f"{current_task} {step_task}"

        # Try RAG retrieval first
        try:
            from app.db.session import async_session_factory
            if async_session_factory is not None:
                from app.services.rag_service import RAGService
                rag_service = RAGService()

                # Extract category filter from query for targeted search
                logger.info(f"[CATEGORY DETECTION] Query: '{search_query}'")
                category_filter = self._extract_category_from_query(search_query)
                logger.info(f"[CATEGORY DETECTION] Detected category: {category_filter}")
                if category_filter:
                    logger.info(f"✓ Will search only {category_filter}_dataset_metadata")
                else:
                    logger.info(f"✗ No category detected - searching all tables")

                logger.info(f"Calling RAG with query: {search_query}")
                retrieval_result = await rag_service.retrieve(
                    query=search_query,
                    top_k=10,
                    category_filter=category_filter,
                )

                logger.info(
                    f"RAG returned {len(retrieval_result.metadata_results)} metadata results, "
                    f"{len(retrieval_result.table_schemas)} table schemas, "
                    f"total_results={retrieval_result.total_results}"
                )

                if retrieval_result.metadata_results:
                    logger.info(
                        f"RAG retrieved {len(retrieval_result.metadata_results)} metadata results "
                        f"with {len(retrieval_result.table_schemas)} table schemas"
                    )

                    # Store retrieval context with table schemas for data loading
                    retrieval_context = {
                        "metadata_results": [
                            {
                                "metadata_id": m.metadata_id,
                                "category": m.category,
                                "file_name": m.file_name,
                                "file_path": m.file_path,  # Include file path for DataFrame loading
                                "table_name": m.table_name,
                                "description": m.description,
                                "summary_text": m.summary_text,  # Rich context about dataset content
                                "columns": m.columns,
                                "row_count": m.row_count,
                                "score": m.score,
                            }
                            for m in retrieval_result.metadata_results
                        ],
                        "table_schemas": [
                            {
                                "table_name": s.table_name,
                                "category": s.category,
                                "description": s.description,
                                "summary_text": s.summary_text,  # Rich context about dataset content
                                "columns": s.columns,
                                "primary_dimensions": s.primary_dimensions,
                                "numeric_columns": s.numeric_columns,
                                "categorical_columns": s.categorical_columns,
                                "row_count": s.row_count,
                                "year_range": s.year_range,
                                "sql_schema_prompt": s.sql_schema_prompt,
                                "file_path": s.file_path,  # Include file path for DataFrame loading
                                "score": s.score,  # Include relevance score
                            }
                            for s in retrieval_result.table_schemas
                        ],
                    }

                    return {
                        "intermediate_results": {
                            **state.get("intermediate_results", {}),
                            "retrieval_context": retrieval_context,
                            "source": "rag",
                        },
                        "retrieval_context": retrieval_context,
                    }

        except Exception as e:
            logger.warning(f"RAG retrieval failed, falling back to file-based: {e}", exc_info=True)

        # Fallback: file-based dataset matching
        return await self._fallback_identify_datasets(state, current_task, step_task, required_data)

    async def _fallback_identify_datasets(
        self, state: GraphState, current_task: str, step_task: str, required_data: List[str]
    ) -> Dict[str, Any]:
        """Fallback: identify datasets using keyword matching on filenames."""
        available_datasets = data_service.get_available_datasets()
        keywords = self._extract_keywords(current_task, step_task, required_data)

        matched_datasets = []
        for ds in available_datasets:
            ds_name_lower = ds["name"].lower()
            ds_path_lower = ds["path"].lower()
            for keyword in keywords:
                if keyword in ds_name_lower or keyword in ds_path_lower:
                    matched_datasets.append(ds)
                    break

        matched_datasets = matched_datasets[:5]

        if not matched_datasets and available_datasets:
            datasets_list = "\n".join([
                f"- {ds['name']}: {ds['path']}"
                for ds in available_datasets[:20]
            ])
            identify_prompt = f"""Given this query, identify the most relevant datasets.

Query: {current_task}
Task: {step_task}

Available datasets:
{datasets_list}

Respond with a JSON list of the most relevant dataset paths:
{{"datasets": ["path1", "path2"]}}"""

            try:
                response = await self._invoke_llm([HumanMessage(content=identify_prompt)])
                parsed = self._parse_json_response(response)
                paths = parsed.get("datasets", [])
                for ds in available_datasets:
                    if ds["path"] in paths:
                        matched_datasets.append(ds)
            except Exception:
                matched_datasets = available_datasets[:2]

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "matched_datasets": matched_datasets,
                "keywords_used": keywords,
                "source": "file",
            },
        }

    async def _load_dataframes_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Load raw DataFrames from files based on RAG retrieval with confidence scoring."""
        retrieval_context = state.get("retrieval_context", {})
        intermediate = state.get("intermediate_results", {})
        source = intermediate.get("source", "file")

        loaded_datasets = {}

        if source == "rag":
            # Load from RAG results with INTELLIGENT SELECTION
            table_schemas = retrieval_context.get("table_schemas", [])

            if not table_schemas:
                logger.warning("No table schemas available for DataFrame loading")
                return {"intermediate_results": {**intermediate, "loaded_datasets": {}}}

            # Get configuration for confidence-based selection
            from app.config import config
            rag_config = config.get_rag_config()
            confidence_threshold = rag_config.get("confidence_threshold", 0.5)
            min_datasets = rag_config.get("min_datasets", 1)
            max_datasets = rag_config.get("max_datasets", 3)

            # Extract scores from table_schemas (scores come from reranker)
            schemas_with_scores = []
            for schema in table_schemas:
                score = schema.get("score", 0.0)
                schemas_with_scores.append((schema, score))

            # Sort by score descending
            schemas_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Select datasets based on confidence
            selected_schemas = []
            for schema, score in schemas_with_scores:
                # Always include top result
                if len(selected_schemas) == 0:
                    selected_schemas.append((schema, score))
                    logger.info(f"✅ Loading top result: {schema.get('table_name')} (score={score:.3f})")
                # Include others only if score is high enough
                elif score >= confidence_threshold and len(selected_schemas) < max_datasets:
                    selected_schemas.append((schema, score))
                    logger.info(f"✅ Loading confident result: {schema.get('table_name')} (score={score:.3f})")
                elif len(selected_schemas) < min_datasets:
                    # Guarantee min_datasets even if below threshold
                    selected_schemas.append((schema, score))
                    logger.info(f"⚠️  Loading fallback result: {schema.get('table_name')} (score={score:.3f} below threshold)")
                else:
                    logger.info(f"❌ Skipping low-confidence result: {schema.get('table_name')} (score={score:.3f})")

            logger.info(
                f"Selected {len(selected_schemas)} datasets out of {len(table_schemas)} candidates. "
                f"Threshold: {confidence_threshold}, Scores: {[f'{s:.3f}' for _, s in selected_schemas]}"
            )

            # Load selected DataFrames with fallback mechanism
            failed_loads = []
            for schema, score in selected_schemas:
                table_name = schema.get("table_name")
                file_path = schema.get("file_path")

                if not file_path:
                    logger.warning(f"No file_path for table {table_name}, skipping DataFrame load")
                    failed_loads.append((table_name, "No file_path"))
                    continue

                try:
                    # Use DataService to load raw DataFrame
                    df = data_service.load_dataset(file_path)

                    # Serialize DataFrame for GraphState
                    loaded_datasets[table_name] = {
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "data": df.to_dict(orient="records"),  # List of dicts
                        "shape": list(df.shape),
                        "metadata": {
                            "table_name": table_name,
                            "category": schema.get("category", ""),
                            "description": schema.get("description", ""),
                            "summary_text": schema.get("summary_text", ""),  # Rich context about dataset
                            "primary_dimensions": schema.get("primary_dimensions", []),
                            "numeric_columns": schema.get("numeric_columns", []),
                            "categorical_columns": schema.get("categorical_columns", []),
                            "year_range": schema.get("year_range", {}),
                            "confidence_score": score,  # Store score
                        },
                        "table_name": table_name,
                        "source": "rag_metadata",  # Changed from 'dataframe' to 'rag_metadata'
                        "file_path": file_path,
                    }

                    logger.info(f"Loaded DataFrame {table_name}: {df.shape} from {file_path}")
                except FileNotFoundError as e:
                    logger.warning(f"⚠️ Dataset file not found: {file_path} (skipping - database may need re-ingestion)")
                    failed_loads.append((table_name, str(e)))
                except Exception as e:
                    logger.error(f"Failed to load DataFrame from {file_path}: {e}", exc_info=True)
                    failed_loads.append((table_name, str(e)))

            # If selected datasets failed but we have more candidates, try loading fallbacks
            if len(loaded_datasets) < min_datasets and failed_loads and len(selected_schemas) < len(table_schemas):
                logger.info(f"Only loaded {len(loaded_datasets)}/{min_datasets} required datasets. Trying fallback datasets...")
                remaining_schemas = [
                    (schema, schema.get("score", 0.0))
                    for schema in table_schemas[len(selected_schemas):max_datasets]
                ]

                for schema, score in remaining_schemas:
                    if len(loaded_datasets) >= min_datasets:
                        break

                    table_name = schema.get("table_name")
                    file_path = schema.get("file_path")

                    if not file_path:
                        continue

                    try:
                        logger.info(f"⚠️ Attempting fallback dataset: {table_name} (score={score:.3f})")
                        df = data_service.load_dataset(file_path)

                        loaded_datasets[table_name] = {
                            "columns": df.columns.tolist(),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                            "data": df.to_dict(orient="records"),
                            "shape": list(df.shape),
                            "metadata": {
                                "table_name": table_name,
                                "category": schema.get("category", ""),
                                "description": schema.get("description", ""),
                                "summary_text": schema.get("summary_text", ""),  # Rich context about dataset
                                "primary_dimensions": schema.get("primary_dimensions", []),
                                "numeric_columns": schema.get("numeric_columns", []),
                                "categorical_columns": schema.get("categorical_columns", []),
                                "year_range": schema.get("year_range", {}),
                                "confidence_score": score,
                            },
                            "table_name": table_name,
                            "source": "rag_metadata",
                            "file_path": file_path,
                        }
                        logger.info(f"✅ Loaded fallback DataFrame {table_name}: {df.shape}")
                    except Exception as e:
                        logger.warning(f"Fallback dataset {table_name} also failed: {e}")

        else:
            # Load from file-based matching
            matched_datasets = intermediate.get("matched_datasets", [])

            for ds in matched_datasets[:3]:  # Load top 3
                try:
                    # Load DataFrame directly
                    df = data_service.load_dataset(ds["path"])

                    # Serialize DataFrame
                    loaded_datasets[ds["name"]] = {
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "data": df.to_dict(orient="records"),
                        "shape": list(df.shape),
                        "metadata": {
                            "category": ds.get("category", ""),
                            "description": ds.get("description", ""),
                        },
                        "source": "file_metadata",  # Changed from 'dataframe' to 'file_metadata'
                        "file_path": ds["path"],
                    }

                    logger.info(f"Loaded DataFrame {ds['name']}: {df.shape} from {ds['path']}")
                except Exception as e:
                    logger.error(f"Failed to load DataFrame from {ds['path']}: {e}", exc_info=True)

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "loaded_datasets": loaded_datasets,
            },
        }


    async def _extract_relevant_data_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Extract relevant data based on query context."""
        try:
            logger.info("EXTRACT_NODE: Starting extraction")
            current_task = state.get("current_task", "")
            loaded_datasets = state.get("intermediate_results", {}).get("loaded_datasets", {})
            retrieval_context = state.get("intermediate_results", {}).get("retrieval_context", {})
            logger.info(f"EXTRACT_NODE: loaded_datasets keys={list(loaded_datasets.keys())}")

            if not loaded_datasets:
                logger.warning("EXTRACT_NODE: No datasets loaded, returning error")
                return {
                    "errors": state.get("errors", []) + ["No datasets loaded"],
                }

            # Build dataset summaries using metadata instead of sample rows
            logger.info("EXTRACT_NODE: Building dataset summaries")
            datasets_summary = []
            for name, data in loaded_datasets.items():
                metadata = data.get("metadata", {})
                metadata_text = ""
                if metadata:
                    # Helper function to extract names from list (handles both str and dict items)
                    def get_names(items):
                        if not items:
                            return []
                        # Handle list of dicts with 'name' key or list of strings
                        return [item.get('name', str(item)) if isinstance(item, dict) else str(item)
                                for item in items]

                    primary_dims = get_names(metadata.get('primary_dimensions', []))
                    numeric_cols = get_names(metadata.get('numeric_columns', []))
                    categorical_cols = get_names(metadata.get('categorical_columns', []))

                    metadata_text = (
                        f"Description: {metadata.get('description', 'N/A')}\n"
                        f"Primary Dimensions: {', '.join(primary_dims)}\n"
                        f"Numeric Columns: {', '.join(numeric_cols)}\n"
                        f"Categorical Columns: {', '.join(categorical_cols)}\n"
                        f"Year Range: {metadata.get('year_range', {})}"
                    )

                summary = (
                    f"Dataset: {name}\n"
                    f"Columns: {', '.join(str(c) for c in data.get('columns', [])[:15])}\n"
                    f"Shape: {data.get('shape', [0, 0])}\n"
                    f"{metadata_text}"
                )
                datasets_summary.append(summary)

            # Include RAG chunk context if available
            chunk_context = ""
            chunks = retrieval_context.get("chunks", [])
            if chunks:
                chunk_texts = [c["chunk_text"] for c in chunks[:5]]
                chunk_context = (
                    "\n\nSemantic context from RAG retrieval:\n"
                    + "\n---\n".join(chunk_texts)
                )

            extraction_prompt = f"""Given this query and available data, identify what to extract.

Query: {current_task}

Available Data:
{chr(10).join(datasets_summary)}
{chunk_context}

Respond in JSON format:
{{
    "relevant_datasets": ["names of relevant datasets"],
    "key_columns": ["columns that answer the query"],
    "summary": "brief description of what data is relevant"
}}"""

            try:
                response = await self._invoke_llm([HumanMessage(content=extraction_prompt)])
                extraction_plan = self._parse_json_response(response)
            except Exception:
                extraction_plan = {
                    "relevant_datasets": list(loaded_datasets.keys()),
                    "key_columns": [],
                    "summary": "All available data",
                }

            # Extract based on plan
            extracted = {}
            for ds_name in extraction_plan.get("relevant_datasets", loaded_datasets.keys()):
                if ds_name in loaded_datasets:
                    ds_data = loaded_datasets[ds_name]
                    key_cols = extraction_plan.get("key_columns", [])

                    if key_cols:
                        available_cols = [c for c in key_cols if c in ds_data.get("columns", [])]
                        if available_cols:
                            filtered_data = [
                                {k: row.get(k) for k in available_cols}
                                for row in ds_data.get("data", [])
                            ]
                            extracted[ds_name] = {
                                **ds_data,
                                "data": filtered_data,
                                "filtered_columns": available_cols,
                            }
                        else:
                            extracted[ds_name] = ds_data
                    else:
                        extracted[ds_name] = ds_data

            logger.info(f"EXTRACT_NODE: Completed, extracted keys={list(extracted.keys())}")
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "extracted_data": extracted,
                    "extraction_plan": extraction_plan,
                },
            }
        except Exception as e:
            logger.error(f"EXTRACT_NODE: Exception occurred: {e}", exc_info=True)
            return {
                "errors": state.get("errors", []) + [f"Extraction failed: {str(e)}"],
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "extracted_data": {},
                },
            }

    async def _format_output_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Format extracted data for the analytics agent."""
        extracted = state.get("intermediate_results", {}).get("extracted_data", {})
        loaded = state.get("intermediate_results", {}).get("loaded_datasets", {})
        source = state.get("intermediate_results", {}).get("source", "file")

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"FORMAT OUTPUT: extracted keys={list(extracted.keys())}, loaded keys={list(loaded.keys())}")

        final_data = extracted if extracted else loaded
        logger.info(f"FORMAT OUTPUT: final_data keys={list(final_data.keys())}")

        summary_parts = []
        for name, data in final_data.items():
            if isinstance(data, dict):
                row_count = len(data.get("data", []))
                col_count = len(data.get("columns", []))
                source_type = data.get("source", "unknown")

                # Indicate whether data comes from SQL results or metadata only
                if row_count > 0:
                    summary_parts.append(f"- {name}: {row_count} rows, {col_count} columns (source: {source_type})")
                else:
                    summary_parts.append(f"- {name}: metadata only, {col_count} columns (awaiting SQL results)")

        summary = "\n".join(summary_parts) if summary_parts else "No data extracted"
        if source == "rag":
            summary = f"[RAG retrieval]\n{summary}"

        logger.info(f"FORMAT OUTPUT: Returning extracted_data with {len(final_data)} datasets")
        return {
            "extracted_data": final_data,
            "current_step": state.get("current_step", 0) + 1,
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "extraction_summary": summary,
            },
            "messages": [
                AIMessage(content=f"Data extracted:\n{summary}")
            ],
        }

    def _extract_category_from_query(self, query: str) -> Optional[str]:
        """Extract category filter from query based on keywords.

        Args:
            query: User query string

        Returns:
            Category name ("employment", "income", "hours_worked") or None for no filter
        """
        query_lower = query.lower()

        # Check for category keywords (order matters - more specific first)
        if any(keyword in query_lower for keyword in [
            "employment", "employed", "unemployment", "job", "labour force", "labor force"
        ]):
            return "employment"
        elif any(keyword in query_lower for keyword in [
            "income", "salary", "wage", "earning", "pay", "compensation"
        ]):
            return "income"
        elif any(keyword in query_lower for keyword in [
            "hours worked", "working hours", "work hours", "overtime"
        ]):
            return "hours_worked"

        # No specific category detected - search all categories
        return None

    def _extract_keywords(
        self,
        task: str,
        step_task: str,
        required_data: List[str],
    ) -> List[str]:
        """Extract search keywords from task and required data."""
        keywords = set()

        for item in required_data:
            keywords.update(item.lower().split())

        for text in [task, step_task]:
            words = text.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    keywords.add(word)

        data_terms = [
            "employment", "income", "labour", "labor", "force",
            "hours", "worked", "salary", "wage", "resident",
            "weather", "temperature", "forecast", "pm2.5", "air",
        ]
        for term in data_terms:
            if term in task.lower() or term in step_task.lower():
                keywords.add(term)

        return list(keywords)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                return json.loads(response[start:end].strip())
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                return json.loads(response[start:end].strip())
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                return json.loads(response[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        return {}

    def _build_response(self, result: GraphState, state: AgentState) -> AgentResponse:
        """Build AgentResponse from graph execution result."""
        extracted_data_dict = result.get("extracted_data", {})
        summary = result.get("intermediate_results", {}).get("extraction_summary", "")
        errors = result.get("errors", [])

        logger.info(f"BUILD RESPONSE: extracted_data keys={list(extracted_data_dict.keys())}, errors={len(errors)}")

        # Convert extracted_data to Pydantic models for type safety
        extracted_datasets = {}
        for dataset_name, dataset_dict in extracted_data_dict.items():
            try:
                # Parse metadata
                metadata_dict = dataset_dict.get("metadata", {})
                year_range_dict = metadata_dict.get("year_range")
                year_range = None
                if year_range_dict and isinstance(year_range_dict, dict):
                    min_year = year_range_dict.get("min")
                    max_year = year_range_dict.get("max")
                    if min_year is not None and max_year is not None:
                        year_range = YearRange(min=min_year, max=max_year)

                # Extract categorical column names from dict structure if needed
                # Database stores [{column, values, cardinality}] but model expects [str]
                categorical_cols = metadata_dict.get("categorical_columns", [])
                if categorical_cols and isinstance(categorical_cols[0], dict):
                    categorical_cols = [col["column"] for col in categorical_cols]

                metadata = DatasetMetadata(
                    table_name=metadata_dict.get("table_name", dataset_name),
                    category=metadata_dict.get("category", ""),
                    description=metadata_dict.get("description", ""),
                    primary_dimensions=metadata_dict.get("primary_dimensions", []),
                    numeric_columns=metadata_dict.get("numeric_columns", []),
                    categorical_columns=categorical_cols,
                    year_range=year_range,
                )

                # Create ExtractedDataset model
                extracted_dataset = ExtractedDataset(
                    path=dataset_dict.get("path", ""),
                    columns=dataset_dict.get("columns", []),
                    shape=tuple(dataset_dict.get("shape", [0, 0])),
                    dtypes=dataset_dict.get("dtypes", {}),
                    data=dataset_dict.get("data", []),
                    metadata=metadata,
                    table_name=dataset_dict.get("table_name", dataset_name),
                    source=dataset_dict.get("source", "file_metadata"),
                )

                extracted_datasets[dataset_name] = extracted_dataset
            except Exception as e:
                logger.warning(f"Failed to convert {dataset_name} to Pydantic model: {e}")
                # Keep the original dict if conversion fails
                continue

        # Create ExtractionResult model
        extraction_result = ExtractionResult(
            extracted_data=extracted_datasets,
            datasets_used=list(extracted_datasets.keys()),
        )

        # Store in state (keep original dict for backward compatibility)
        state.extracted_data = extracted_data_dict
        state.errors.extend(errors)

        return AgentResponse(
            success=len(extracted_datasets) > 0 and len(errors) == 0,
            message=f"Data extraction complete.\n{summary}" if summary else "Data extraction complete.",
            data=extraction_result.model_dump(),  # Convert to dict for GraphState
            next_agent=AgentRole.ANALYTICS,
            state=state,
        )
