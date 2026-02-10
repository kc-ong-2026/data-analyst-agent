"""Data Extraction Agent - Extracts data using LangGraph workflow with RAG."""

import json
import logging
from typing import Any, Dict, List

from sqlalchemy import text
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
                logger.info(f"Calling RAG with query: {search_query}")
                retrieval_result = await rag_service.retrieve(
                    query=search_query,
                    top_k=10,
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
        """Node: Generate SQL queries for data retrieval.

        Default behavior: Always generate SQL queries to retrieve data from tables.
        This ensures all data queries get proper filtering/aggregation.
        """
        current_task = state.get("current_task", "")
        retrieval_context = state.get("retrieval_context", {})

        # Get table schemas from retrieval context
        table_schemas = retrieval_context.get("table_schemas", [])
        if not table_schemas:
            logger.info("No table schemas available for SQL generation")
            return {"sql_queries": [], "sql_results": {}}

        # Generate SQL query using LLM (single table only)
        try:
            sql_queries = await self._generate_sql_with_llm(current_task, table_schemas)
            if sql_queries:
                logger.info(f"Generated SQL query for single table analysis")
                logger.info(f"📝 Generated SQL: {sql_queries[0].get('sql', '')}")
                logger.info(f"   Description: {sql_queries[0].get('description', '')}")

            # Validate SQL query for safety
            validated_queries = self._validate_sql_queries(sql_queries, table_schemas)
            if not validated_queries:
                logger.warning("No valid SQL query after validation")
                return {"sql_queries": [], "sql_results": {}}

            # Log the validated query
            table_name = validated_queries[0].get("table_name", "unknown")
            validated_sql = validated_queries[0].get("sql", "")
            logger.info(f"✅ Validated SQL query for table: {table_name}")
            logger.info(f"   SQL: {validated_sql}")

            # Execute SQL query
            sql_results = await self._execute_sql_queries(validated_queries)
            total_rows = sum(len(rows) for rows in sql_results.values())
            logger.info(f"Executed SQL query, got {total_rows} rows from {table_name}")

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
        """Detect if query requires SQL aggregation or filtering.

        Args:
            query: User query

        Returns:
            True if query requires SQL (aggregation, filtering, or grouping)
        """
        query_lower = query.lower()

        # Aggregation keywords
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
            "rate",  # employment rate, unemployment rate, etc.
        ]

        # Grouping/dimension keywords (indicates need for GROUP BY)
        grouping_keywords = [
            "by industry", "by sector", "by age", "by sex",
            "by qualification", "by education", "by gender",
            "by occupation", "by region", "by area",
        ]

        # Year range patterns (indicates need for WHERE filtering)
        year_range_patterns = [
            r'\b(19\d{2}|20[0-4]\d)\s*(?:to|-|through|until|and)\s*(19\d{2}|20[0-4]\d)\b',  # 2020-2025, 2020 to 2025
            r'\bbetween\s+(19\d{2}|20[0-4]\d)\s+(?:and|to)\s+(19\d{2}|20[0-4]\d)\b',  # between 2020 and 2025
            r'\bfrom\s+(19\d{2}|20[0-4]\d)\s+(?:to|until)\s+(19\d{2}|20[0-4]\d)\b',  # from 2020 to 2025
        ]

        # Action verbs indicating data retrieval (usually needs SQL)
        action_keywords = [
            "show", "display", "get", "find", "retrieve",
            "list", "what is", "what are", "how many",
        ]

        # Check aggregation keywords
        if any(keyword in query_lower for keyword in aggregation_keywords):
            return True

        # Check grouping keywords
        if any(keyword in query_lower for keyword in grouping_keywords):
            return True

        # Check for year ranges (indicates filtering needed)
        import re
        for pattern in year_range_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check action verbs (most queries with these need SQL)
        if any(keyword in query_lower for keyword in action_keywords):
            return True

        # Default: if query contains numbers (years, values), likely needs SQL
        if re.search(r'\b(19\d{2}|20[0-4]\d)\b', query_lower):
            return True

        return False

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
        # Build schema descriptions for LLM (single table analysis)
        schema_prompts = []
        for schema in table_schemas[:3]:  # Limit to top 3 most relevant tables
            schema_prompt = schema.get("sql_schema_prompt", "")
            if schema_prompt:
                schema_prompts.append(schema_prompt)

        schemas_text = "\n\n".join(schema_prompts)

        sql_prompt = f"""Generate a SINGLE SQL query to answer the user's question using ONE table from the provided schemas.

User Question: {query}

Available Tables:
{schemas_text}

Requirements:
1. Identify the MOST RELEVANT single table that best answers the question
2. Generate ONE SQL query that queries ONLY that table
3. **CRITICAL**: Use SELECT * to retrieve ALL columns from filtered rows
4. Use proper PostgreSQL syntax
5. Include appropriate WHERE clauses for filtering
6. Add ORDER BY for trends/sorting (use columns that exist in the table)
7. Include LIMIT to prevent large result sets (default 500 rows)
8. **CRITICAL**: ONLY use column names in WHERE/ORDER BY that are EXPLICITLY LISTED in the "Columns:" section
9. Do NOT use JOINs - query only ONE table
10. Do NOT use GROUP BY or aggregations - return raw filtered rows with all columns

CRITICAL - Column Validation:
- Before using ANY column in your SQL query, VERIFY it exists in the "Columns:" list above
- If a column doesn't exist (e.g., "sex" column when only "industry" exists), DO NOT use it
- It's better to return broader results than to fail with "column does not exist" error

CRITICAL - Handling Dimensions (age, sex, industry, qualification, etc.):
- **ALWAYS use SELECT * to get all columns** - do NOT select specific columns
- If the question does NOT specify a dimension value (e.g., "employment rate" without mentioning age/sex/industry):
  * Return ALL rows without filtering by that dimension
  * Example: If table has "age", "sex", "employment_count" and question doesn't specify age:
    → SELECT * FROM table WHERE year BETWEEN 2020 AND 2023 ORDER BY year LIMIT 500
- If the question DOES specify a dimension value (e.g., "employment rate for ages 25-54" OR "technology sector"):
  * **CRITICAL**: Filter to the SINGLE MOST RELEVANT value for that dimension
  * Use WHERE clause to filter: WHERE industry LIKE '%technology%' OR industry LIKE '%IT%'
  * **For industry/sector questions**: Pick the ONE most representative industry value (e.g., "IT and other information services" for "technology sector")
  * Do NOT return multiple industries - filter to just ONE that best matches the user's intent
  * Use pattern matching (LIKE '%keyword%') to find the most relevant industry
- **ALWAYS use ORDER BY year** when time series data is requested
- Check categorical column values to match user's intent (e.g., "Total" rows often exist)
- Common dimension columns: age, sex, industry, qualification, employment_status
- Do NOT use GROUP BY or aggregation functions - return raw filtered rows

Examples:
1. "Employment rate from 2019-2021" (no age/sex/industry specified):
   → SELECT * FROM table WHERE year BETWEEN 2019 AND 2021 ORDER BY year LIMIT 500

2. "Technology sector employment from 2022-2023" (specific industry + years):
   → SELECT * FROM table
      WHERE (industry2 LIKE '%IT%' OR industry2 LIKE '%information%')
        AND year BETWEEN 2022 AND 2023
      ORDER BY year LIMIT 10
   → Pick ONLY the most relevant single industry (e.g., "IT and other information services")

3. "Female employment rate in 2020":
   → SELECT * FROM table
      WHERE (sex = 'Female' OR sex LIKE '%Female%') AND year = 2020
      ORDER BY year LIMIT 500

4. "Finance sector from 2022-2023":
   → SELECT * FROM table
      WHERE (industry2 LIKE '%finance%' OR industry2 LIKE '%financial%')
        AND year BETWEEN 2022 AND 2023
      ORDER BY year LIMIT 10

Respond in JSON format:
{{
    "query": {{
        "sql": "SELECT * FROM table_name WHERE ... ORDER BY ... LIMIT ...",
        "description": "what this query does",
        "table_name": "the_single_table_name"
    }}
}}

IMPORTANT:
- Only use SELECT statements. Do not use INSERT, UPDATE, DELETE, DROP, or other modification commands.
- Query ONLY ONE table - the most relevant one for answering the question.
- Return ONE query object, not an array of queries.
- Use SELECT * to retrieve all columns from filtered rows.
- Focus filtering in WHERE clause, not on column selection."""

        response = await self._invoke_llm([HumanMessage(content=sql_prompt)])
        parsed = self._parse_json_response(response)

        # Handle single query format (new simplified approach)
        query_obj = parsed.get("query", {})
        if query_obj and query_obj.get("sql"):
            # Ensure table_name is set
            if not query_obj.get("table_name"):
                query_obj["table_name"] = "unknown"
            return [query_obj]  # Return as list for compatibility

        # Fallback to old format if needed
        queries = parsed.get("queries", [])
        return queries if queries else []

    def _validate_sql_queries(
        self, queries: List[Dict[str, Any]], table_schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate SQL queries for safety and column existence.

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
                sql += " LIMIT 500"
                query_dict["sql"] = sql

            validated.append(query_dict)

        return validated

    async def _execute_sql_queries(
        self, queries: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute validated SQL query (single table only).

        Args:
            queries: List with single SQL query dictionary

        Returns:
            Dictionary with single key "query_result" containing all result rows
        """
        from app.db.session import get_db

        if not queries:
            return {}

        # Execute the single table query
        async with get_db() as session:
            query_dict = queries[0]  # Only one query for single table
            sql = query_dict.get("sql", "")
            table_name = query_dict.get("table_name", "")
            description = query_dict.get("description", "Query result")

            try:
                # Execute query with text() for parameterized execution
                logger.info(f"🔍 Executing SQL on table '{table_name}':")
                logger.info(f"   {sql}")
                result = await session.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()

                # Convert to list of dicts
                result_data = [
                    {col: val for col, val in zip(columns, row)}
                    for row in rows
                ]

                logger.info(
                    f"✅ SQL query returned {len(result_data)} rows from table: {table_name}"
                )

                # Return single result with table name for context
                return {"query_result": result_data}

            except Exception as e:
                logger.error(f"Failed to execute SQL query on {table_name}: {e}", exc_info=True)
                return {}

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

        # Merge SQL results if available (single table query result)
        sql_results = state.get("sql_results", {})
        sql_queries = state.get("sql_queries", [])
        if sql_results and "query_result" in sql_results and sql_queries:
            loaded_datasets = result.get("intermediate_results", {}).get("loaded_datasets", {})
            rows = sql_results["query_result"]

            # Get the table name from the SQL query
            table_name = sql_queries[0].get("table_name", "") if sql_queries else ""

            if rows and table_name:
                # Merge SQL results into the corresponding table entry
                if table_name in loaded_datasets:
                    loaded_datasets[table_name]["data"] = rows
                    loaded_datasets[table_name]["shape"] = [len(rows), len(loaded_datasets[table_name].get("columns", []))]
                    loaded_datasets[table_name]["source"] = "sql"
                    logger.info(f"Merged SQL query result ({len(rows)} rows) into {table_name}")
                else:
                    # Table not in loaded_datasets, create new entry with SQL results
                    columns = list(rows[0].keys()) if rows else []
                    loaded_datasets[table_name] = {
                        "path": "SQL query result",
                        "columns": columns,
                        "shape": [len(rows), len(columns)],
                        "dtypes": {},
                        "data": rows,
                        "source": "sql",
                    }
                    logger.info(f"Created new entry {table_name} with SQL query result ({len(rows)} rows)")

                result["intermediate_results"]["loaded_datasets"] = loaded_datasets

        return result

    async def _load_from_rag(self, state: GraphState) -> Dict[str, Any]:
        """Load datasets metadata from RAG retrieval results.

        Single-table analysis: Only load metadata from the most relevant table.
        Sample rows are NOT loaded - SQL query results will provide the actual data.
        """
        retrieval_context = state.get("intermediate_results", {}).get("retrieval_context", {})
        metadata_results = retrieval_context.get("metadata_results", [])
        table_schemas = retrieval_context.get("table_schemas", [])

        loaded_datasets = {}

        # Load metadata only (no sample rows) from the MOST RELEVANT table
        for metadata_info in metadata_results[:1]:  # Only load from top 1 most relevant table
            table_name = metadata_info.get("table_name", "")
            if not table_name:
                continue

            try:
                # Get column info
                columns = [c["name"] for c in metadata_info.get("columns", [])]

                # Find corresponding table schema for richer metadata
                table_schema = None
                for schema in table_schemas:
                    if schema.get("table_name") == table_name:
                        table_schema = schema
                        break

                # Store metadata information without sample rows
                loaded_datasets[table_name] = {
                    "path": metadata_info.get("file_name", ""),
                    "columns": columns,
                    "shape": [metadata_info.get("row_count", 0), len(columns)],
                    "dtypes": {c["name"]: c.get("dtype", "object") for c in metadata_info.get("columns", [])},
                    "data": [],  # Empty - will use SQL results
                    "metadata": {
                        "table_name": table_name,
                        "category": metadata_info.get("category", ""),
                        "description": metadata_info.get("description", "") if not table_schema else table_schema.get("description", ""),
                        "primary_dimensions": table_schema.get("primary_dimensions", []) if table_schema else [],
                        "numeric_columns": table_schema.get("numeric_columns", []) if table_schema else [],
                        "categorical_columns": table_schema.get("categorical_columns", []) if table_schema else [],
                        "year_range": table_schema.get("year_range", {}) if table_schema else {},
                    },
                    "table_name": table_name,
                    "source": "rag_metadata",
                }

                logger.info(f"Loaded metadata for {table_name} (no sample rows - will use SQL results)")
            except Exception as e:
                logger.warning(f"Failed to load metadata from {table_name}: {e}")

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "loaded_datasets": loaded_datasets,
            },
        }

    async def _load_from_files(self, state: GraphState) -> Dict[str, Any]:
        """Load dataset metadata from file system (original flow).

        Sample rows are NOT loaded - SQL query results will provide the actual data.
        """
        matched_datasets = state.get("intermediate_results", {}).get("matched_datasets", [])

        loaded_datasets = {}
        load_errors = []

        for ds in matched_datasets:
            try:
                info = data_service.get_dataset_info(ds["path"])
                # Do NOT load sample data - only metadata
                loaded_datasets[ds["name"]] = {
                    "path": ds["path"],
                    "columns": info.get("columns", []),
                    "shape": info.get("shape", [0, 0]),
                    "dtypes": info.get("dtypes", {}),
                    "data": [],  # Empty - will use SQL results
                    "metadata": {
                        "description": info.get("description", ""),
                        "category": ds.get("category", ""),
                    },
                    "source": "file_metadata",
                }
                logger.info(f"Loaded metadata for {ds['name']} (no sample rows - will use SQL results)")
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

                metadata = DatasetMetadata(
                    table_name=metadata_dict.get("table_name", dataset_name),
                    category=metadata_dict.get("category", ""),
                    description=metadata_dict.get("description", ""),
                    primary_dimensions=metadata_dict.get("primary_dimensions", []),
                    numeric_columns=metadata_dict.get("numeric_columns", []),
                    categorical_columns=metadata_dict.get("categorical_columns", []),
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
