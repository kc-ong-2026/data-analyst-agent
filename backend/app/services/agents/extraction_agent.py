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
        retrieve_context -> load_raw_data -> extract_relevant_data -> format_output -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("load_raw_data", self._load_raw_data_node)
        workflow.add_node("extract_relevant_data", self._extract_relevant_data_node)
        workflow.add_node("format_output", self._format_output_node)

        # Set entry point
        workflow.set_entry_point("retrieve_context")

        # Add edges with conditional routing
        workflow.add_edge("retrieve_context", "load_raw_data")
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

                if retrieval_result.chunks:
                    logger.info(
                        f"RAG retrieved {len(retrieval_result.chunks)} chunks "
                        f"from {len(retrieval_result.datasets)} datasets"
                    )

                    # Store retrieval context
                    retrieval_context = {
                        "chunks": [
                            {
                                "chunk_id": c.chunk_id,
                                "dataset_name": c.dataset_name,
                                "chunk_text": c.chunk_text,
                                "group_key": c.group_key,
                                "score": c.score,
                            }
                            for c in retrieval_result.chunks
                        ],
                        "datasets": [
                            {
                                "dataset_name": d.dataset_name,
                                "file_path": d.file_path,
                                "category": d.category,
                                "description": d.description,
                                "columns": d.columns,
                                "row_count": d.row_count,
                                "year_range": d.year_range,
                            }
                            for d in retrieval_result.datasets
                        ],
                        "raw_data": retrieval_result.raw_data,
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
            logger.warning(f"RAG retrieval failed, falling back to file-based: {e}")

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

    async def _load_raw_data_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Load data from RAG results or fall back to file loading."""
        intermediate = state.get("intermediate_results", {})
        source = intermediate.get("source", "file")

        if source == "rag":
            return await self._load_from_rag(state)
        else:
            return await self._load_from_files(state)

    async def _load_from_rag(self, state: GraphState) -> Dict[str, Any]:
        """Load datasets from RAG retrieval results."""
        retrieval_context = state.get("intermediate_results", {}).get("retrieval_context", {})
        raw_data = retrieval_context.get("raw_data", {})
        datasets = retrieval_context.get("datasets", [])
        chunks = retrieval_context.get("chunks", [])

        loaded_datasets = {}

        # Group raw data by dataset name
        for dataset_info in datasets:
            ds_name = dataset_info["dataset_name"]

            # Collect all raw data rows for this dataset
            all_rows = []
            for key, rows in raw_data.items():
                if key.startswith(ds_name):
                    all_rows.extend(rows)

            # Get column info
            columns = [c["name"] for c in dataset_info.get("columns", [])]

            loaded_datasets[ds_name] = {
                "path": dataset_info.get("file_path", ""),
                "columns": columns,
                "shape": [len(all_rows), len(columns)],
                "dtypes": {c["name"]: c.get("dtype", "object") for c in dataset_info.get("columns", [])},
                "data": all_rows if all_rows else [],
                "rag_chunks": [
                    c for c in chunks if c["dataset_name"] == ds_name
                ],
            }

        # If RAG didn't return raw data, try loading from files
        if not loaded_datasets:
            for dataset_info in datasets:
                try:
                    file_path = dataset_info.get("file_path", "")
                    if file_path:
                        info = data_service.get_dataset_info(file_path)
                        sample_data = data_service.query_dataset(file_path, limit=50)
                        loaded_datasets[dataset_info["dataset_name"]] = {
                            "path": file_path,
                            "columns": info.get("columns", []),
                            "shape": info.get("shape", [0, 0]),
                            "dtypes": info.get("dtypes", {}),
                            "data": sample_data,
                        }
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_info['dataset_name']} from file: {e}")

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
