"""Data Extraction Agent - Extracts data using LangGraph workflow with RAG."""

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from .base_agent import (
    AgentRole,
    AgentResponse,
    AgentState,
    BaseAgent,
    GraphState,
)
from app.services.data_service import data_service

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
        return """You are a Data Extraction Agent for Singapore government data.

Your responsibilities:
1. Identify and access relevant datasets based on the task
2. Extract specific data points or ranges as requested
3. Format extracted data for the Analytics agent
4. Handle data quality issues and missing values

Always be precise about which data you're extracting and from which source."""

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for data extraction.

        Flow:
        retrieve_context -> generate_sql_queries -> load_raw_data -> extract_relevant_data -> format_output -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_sql_queries", self._generate_sql_queries_node)
        workflow.add_node("load_raw_data", self._load_raw_data_node)
        workflow.add_node("extract_relevant_data", self._extract_relevant_data_node)
        workflow.add_node("format_output", self._format_output_node)

        # Set entry point
        workflow.set_entry_point("retrieve_context")

        # Add edges with conditional routing
        workflow.add_edge("retrieve_context", "generate_sql_queries")
        workflow.add_edge("generate_sql_queries", "load_raw_data")
        workflow.add_conditional_edges(
            "load_raw_data",
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
        if loaded_data:
            return "extract"
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
                retrieval_result = await rag_service.retrieve(
                    query=search_query,
                    top_k=10,
                )

                if retrieval_result.metadata_results:
                    logger.info(
                        f"RAG retrieved {len(retrieval_result.metadata_results)} metadata results "
                        f"with {len(retrieval_result.table_schemas)} table schemas"
                    )

                    # Store retrieval context with table schemas for SQL generation
                    retrieval_context = {
                        "metadata_results": [
                            {
                                "metadata_id": m.metadata_id,
                                "category": m.category,
                                "file_name": m.file_name,
                                "table_name": m.table_name,
                                "description": m.description,
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
                                "columns": s.columns,
                                "primary_dimensions": s.primary_dimensions,
                                "numeric_columns": s.numeric_columns,
                                "categorical_columns": s.categorical_columns,
                                "row_count": s.row_count,
                                "year_range": s.year_range,
                                "sql_schema_prompt": s.sql_schema_prompt,
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

    async def _generate_sql_queries_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate SQL queries for aggregation if needed."""
        current_task = state.get("current_task", "")
        retrieval_context = state.get("retrieval_context", {})

        # Check if SQL aggregation is needed
        if not self._requires_sql_aggregation(current_task):
            logger.info("Query does not require SQL aggregation, skipping")
            return {"sql_queries": [], "sql_results": {}}

        # Get table schemas from retrieval context
        table_schemas = retrieval_context.get("table_schemas", [])
        if not table_schemas:
            logger.info("No table schemas available for SQL generation")
            return {"sql_queries": [], "sql_results": {}}

        # Generate SQL queries using LLM
        try:
            sql_queries = await self._generate_sql_with_llm(current_task, table_schemas)
            logger.info(f"Generated {len(sql_queries)} SQL queries")

            # Validate SQL queries for safety
            validated_queries = self._validate_sql_queries(sql_queries, table_schemas)
            logger.info(f"Validated {len(validated_queries)} SQL queries")

            # Execute SQL queries
            sql_results = await self._execute_sql_queries(validated_queries)
            logger.info(f"Executed SQL queries, got {len(sql_results)} result sets")

            return {
                "sql_queries": validated_queries,
                "sql_results": sql_results,
            }
        except Exception as e:
            logger.error(f"SQL generation/execution failed: {e}", exc_info=True)
            return {
                "sql_queries": [],
                "sql_results": {},
                "errors": state.get("errors", []) + [f"SQL generation failed: {str(e)}"],
            }

    def _requires_sql_aggregation(self, query: str) -> bool:
        """Detect if query requires SQL aggregation.

        Args:
            query: User query

        Returns:
            True if query contains aggregation keywords
        """
        aggregation_keywords = [
            "average", "avg", "mean",
            "total", "sum",
            "count", "number of",
            "trend", "over time", "by year", "by month",
            "group by", "grouped by",
            "maximum", "max", "highest",
            "minimum", "min", "lowest",
            "median",
            "compare", "comparison",
            "distribution",
            "percentage", "percent",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in aggregation_keywords)

    async def _generate_sql_with_llm(
        self, query: str, table_schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate SQL queries using LLM with table schemas.

        Args:
            query: User query
            table_schemas: List of table schema dictionaries

        Returns:
            List of SQL query dictionaries with {sql, table_name, description}
        """
        # Build schema descriptions for LLM
        schema_prompts = []
        for schema in table_schemas[:3]:  # Limit to top 3 most relevant tables
            schema_prompt = schema.get("sql_schema_prompt", "")
            if schema_prompt:
                schema_prompts.append(schema_prompt)

        schemas_text = "\n\n".join(schema_prompts)

        sql_prompt = f"""Generate SQL queries to answer the user's question using the provided table schemas.

User Question: {query}

Available Tables:
{schemas_text}

Requirements:
1. Generate 1-3 SQL queries that answer the question
2. Use proper PostgreSQL syntax
3. Include appropriate WHERE clauses for filtering
4. Use GROUP BY for aggregations
5. Add ORDER BY for trends/sorting
6. Include LIMIT 100 to prevent large result sets
7. Use column names exactly as shown in the schema
8. Only query the tables provided above

Respond in JSON format:
{{
    "queries": [
        {{
            "sql": "SELECT ... FROM ...",
            "table_name": "table_name",
            "description": "what this query does"
        }}
    ]
}}

IMPORTANT: Only use SELECT statements. Do not use INSERT, UPDATE, DELETE, DROP, or other modification commands."""

        response = await self._invoke_llm([HumanMessage(content=sql_prompt)])
        parsed = self._parse_json_response(response)

        queries = parsed.get("queries", [])
        return queries

    def _validate_sql_queries(
        self, queries: List[Dict[str, Any]], table_schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate SQL queries for safety.

        Args:
            queries: List of SQL query dictionaries
            table_schemas: Available table schemas

        Returns:
            List of validated SQL queries
        """
        validated = []
        valid_tables = {schema.get("table_name") for schema in table_schemas}

        # Dangerous SQL keywords
        dangerous_keywords = [
            "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER",
            "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "--", ";--"
        ]

        for query_dict in queries:
            sql = query_dict.get("sql", "")
            table_name = query_dict.get("table_name", "")

            if not sql or not table_name:
                logger.warning("Skipping query with missing sql or table_name")
                continue

            # Check for dangerous keywords
            sql_upper = sql.upper()
            if any(keyword in sql_upper for keyword in dangerous_keywords):
                logger.warning(f"Skipping query with dangerous keyword: {sql[:100]}")
                continue

            # Check table name is in valid list
            if table_name not in valid_tables:
                logger.warning(f"Skipping query for unknown table: {table_name}")
                continue

            # Ensure LIMIT is present
            if "LIMIT" not in sql_upper:
                sql += " LIMIT 100"
                query_dict["sql"] = sql

            validated.append(query_dict)

        return validated

    async def _execute_sql_queries(
        self, queries: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute validated SQL queries.

        Args:
            queries: List of validated SQL query dictionaries

        Returns:
            Dictionary mapping table names to result rows
        """
        from app.db.session import get_db

        results = {}

        async with get_db() as session:
            for query_dict in queries:
                sql = query_dict.get("sql", "")
                table_name = query_dict.get("table_name", "")
                description = query_dict.get("description", "")

                try:
                    # Execute query with text() for parameterized execution
                    result = await session.execute(text(sql))
                    rows = result.fetchall()
                    columns = result.keys()

                    # Convert to list of dicts
                    result_data = [
                        {col: val for col, val in zip(columns, row)}
                        for row in rows
                    ]

                    # Store results with descriptive key
                    result_key = f"{table_name}_{description}" if description else table_name
                    results[result_key] = result_data

                    logger.info(f"SQL query returned {len(result_data)} rows for {result_key}")
                except Exception as e:
                    logger.error(f"Failed to execute SQL for {table_name}: {e}")

        return results

    async def _load_raw_data_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Load data from RAG results or fall back to file loading.

        Merges SQL query results if available.
        """
        intermediate = state.get("intermediate_results", {})
        source = intermediate.get("source", "file")

        # Load data based on source
        if source == "rag":
            result = await self._load_from_rag(state)
        else:
            result = await self._load_from_files(state)

        # Merge SQL results if available
        sql_results = state.get("sql_results", {})
        if sql_results:
            loaded_datasets = result.get("intermediate_results", {}).get("loaded_datasets", {})

            # Add SQL results as separate datasets
            for result_key, rows in sql_results.items():
                if rows:
                    # Extract columns from first row
                    columns = list(rows[0].keys()) if rows else []
                    loaded_datasets[f"SQL_{result_key}"] = {
                        "path": "SQL query result",
                        "columns": columns,
                        "shape": [len(rows), len(columns)],
                        "dtypes": {},
                        "data": rows,
                        "source": "sql",
                    }

            result["intermediate_results"]["loaded_datasets"] = loaded_datasets
            logger.info(f"Merged {len(sql_results)} SQL result sets into loaded datasets")

        return result

    async def _load_from_rag(self, state: GraphState) -> Dict[str, Any]:
        """Load datasets from RAG retrieval results.

        In folder-based architecture, we rely on SQL queries for data loading
        rather than loading full datasets into memory.
        """
        retrieval_context = state.get("intermediate_results", {}).get("retrieval_context", {})
        metadata_results = retrieval_context.get("metadata_results", [])

        loaded_datasets = {}

        # Load sample data from data tables for preview/context
        from app.db.session import get_db

        async with get_db() as session:
            for metadata_info in metadata_results[:3]:  # Limit to top 3 results
                table_name = metadata_info.get("table_name", "")
                if not table_name:
                    continue

                try:
                    # Load sample rows for context
                    sample_sql = text(f'SELECT * FROM "{table_name}" LIMIT 50')
                    result = await session.execute(sample_sql)
                    rows = result.fetchall()
                    columns_list = result.keys()

                    sample_data = [
                        {col: val for col, val in zip(columns_list, row)}
                        for row in rows
                    ]

                    # Get column info
                    columns = [c["name"] for c in metadata_info.get("columns", [])]

                    loaded_datasets[table_name] = {
                        "path": metadata_info.get("file_name", ""),
                        "columns": columns,
                        "shape": [metadata_info.get("row_count", 0), len(columns)],
                        "dtypes": {c["name"]: c.get("dtype", "object") for c in metadata_info.get("columns", [])},
                        "data": sample_data,
                        "table_name": table_name,
                        "source": "rag_folder",
                    }

                    logger.info(f"Loaded {len(sample_data)} sample rows from {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to load sample data from {table_name}: {e}")

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "loaded_datasets": loaded_datasets,
            },
        }

    async def _load_from_files(self, state: GraphState) -> Dict[str, Any]:
        """Load datasets from file system (original flow)."""
        matched_datasets = state.get("intermediate_results", {}).get("matched_datasets", [])

        loaded_datasets = {}
        load_errors = []

        for ds in matched_datasets:
            try:
                info = data_service.get_dataset_info(ds["path"])
                sample_data = data_service.query_dataset(ds["path"], limit=50)
                loaded_datasets[ds["name"]] = {
                    "path": ds["path"],
                    "columns": info.get("columns", []),
                    "shape": info.get("shape", [0, 0]),
                    "dtypes": info.get("dtypes", {}),
                    "data": sample_data,
                }
            except Exception as e:
                load_errors.append(f"Failed to load {ds['name']}: {str(e)}")

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "loaded_datasets": loaded_datasets,
                "load_errors": load_errors,
            },
        }

    async def _extract_relevant_data_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Extract relevant data based on query context."""
        current_task = state.get("current_task", "")
        loaded_datasets = state.get("intermediate_results", {}).get("loaded_datasets", {})
        retrieval_context = state.get("intermediate_results", {}).get("retrieval_context", {})

        if not loaded_datasets:
            return {
                "errors": state.get("errors", []) + ["No datasets loaded"],
            }

        # Build dataset summaries
        datasets_summary = []
        for name, data in loaded_datasets.items():
            summary = (
                f"Dataset: {name}\n"
                f"Columns: {', '.join(str(c) for c in data.get('columns', [])[:15])}\n"
                f"Shape: {data.get('shape', [0, 0])}\n"
                f"Sample row: {json.dumps(data['data'][0] if data.get('data') else {}, default=str)[:200]}"
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

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "extracted_data": extracted,
                "extraction_plan": extraction_plan,
            },
        }

    async def _format_output_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Format extracted data for the analytics agent."""
        extracted = state.get("intermediate_results", {}).get("extracted_data", {})
        loaded = state.get("intermediate_results", {}).get("loaded_datasets", {})
        source = state.get("intermediate_results", {}).get("source", "file")

        final_data = extracted if extracted else loaded

        summary_parts = []
        for name, data in final_data.items():
            if isinstance(data, dict):
                row_count = len(data.get("data", []))
                col_count = len(data.get("columns", []))
                summary_parts.append(f"- {name}: {row_count} rows, {col_count} columns")

        summary = "\n".join(summary_parts) if summary_parts else "No data extracted"
        if source == "rag":
            summary = f"[RAG retrieval]\n{summary}"

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
        extracted_data = result.get("extracted_data", {})
        summary = result.get("intermediate_results", {}).get("extraction_summary", "")
        errors = result.get("errors", [])

        state.extracted_data = extracted_data
        state.errors.extend(errors)

        return AgentResponse(
            success=len(extracted_data) > 0 or len(errors) == 0,
            message=f"Data extraction complete.\n{summary}" if summary else "Data extraction complete.",
            data={
                "extracted_data": extracted_data,
                "datasets_used": list(extracted_data.keys()),
            },
            next_agent=AgentRole.ANALYTICS,
            state=state,
        )
