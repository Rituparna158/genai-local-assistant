"""
SQL Service Module.
Coordinates Schema-Aware Text-to-SQL generation, safety validation,
database execution, self-healing retries, query logging, and natural language synthesis.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.api.ollama_service import ollama_service
from src.sql_agent.db_connector import db_connector
from src.sql_agent.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class SQLService:
    """Enterprise Text-to-SQL execution engine with guardrails and self-healing retries."""

    def __init__(self):
        self.db = db_connector
        self.llm = ollama_service.llm

        # Prompt 1: Text-to-SQL Generator Prompt
        self.sql_gen_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert PostgreSQL data engineer.\n"
                "Your task is to convert the user's natural language question into a valid, efficient SQL SELECT query.\n\n"
                "DATABASE SCHEMA:\n{schema_context}\n\n"
                "RULES:\n"
                "1. Generate ONLY valid PostgreSQL SELECT statements. Never write DROP, DELETE, UPDATE, INSERT, or ALTER.\n"
                "2. Use table aliases where appropriate (e.g., customers c, orders o, products p).\n"
                "3. Apply filters dynamically in the WHERE clause using standard SQL operators (=, BETWEEN, >=, <=, IN, ILIKE) according to the user's question.\n"
                "4. When counting records, use COUNT(*) with appropriate WHERE filters. Avoid using CASE expressions inside COUNT() to prevent non-null zero-counting errors.\n"
                "5. When aggregating metrics (e.g. revenue, totals), use SUM(), AVG(), or MIN()/MAX() with appropriate GROUP BY clauses.\n"
                "6. Return ONLY the raw SQL statement. Do not add markdown fences, explanations, or quotes.\n",
            ),
            ("human", "{question}"),
        ])

        # Prompt 2: Self-Healing Correction Prompt
        self.sql_fix_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a PostgreSQL database specialist.\n"
                "The previous SQL query you generated produced an execution error.\n"
                "DATABASE SCHEMA:\n{schema_context}\n\n"
                "PREVIOUS QUERY:\n{failed_query}\n\n"
                "DATABASE ERROR:\n{error_message}\n\n"
                "Fix the error and return ONLY the corrected, valid PostgreSQL SELECT statement.\n",
            ),
            ("human", "Fix the SQL query to answer: {question}"),
        ])

        # Prompt 3: Result Synthesis Prompt (Formats raw rows into natural language)
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an insightful data analyst.\n"
                "Based on the user's question and the SQL query results from the database, "
                "provide a clear, direct, and professional answer summarizing the findings.\n"
                "Include key numbers, names, and metrics from the results.\n\n"
                "DATABASE RESULTS:\n{query_results}\n",
            ),
            ("human", "{question}"),
        ])

    async def _generate_sql(self, question: str, schema_context: str) -> str:
        """Invokes LLM to generate raw SQL from natural language question."""
        prompt = self.sql_gen_prompt.invoke({
            "schema_context": schema_context,
            "question": question,
        })
        response = await self.llm.ainvoke(prompt)
        return str(response.content).strip()

    async def _fix_sql(self, question: str, schema_context: str, failed_query: str, error_message: str) -> str:
        """Self-healing retry: asks LLM to fix a failed SQL statement."""
        prompt = self.sql_fix_prompt.invoke({
            "schema_context": schema_context,
            "failed_query": failed_query,
            "error_message": error_message,
            "question": question,
        })
        response = await self.llm.ainvoke(prompt)
        return str(response.content).strip()

    async def _synthesize_answer(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Converts raw tabular query results into natural language text."""
        prompt = self.synthesis_prompt.invoke({
            "query_results": str(results),
            "question": question,
        })
        response = await self.llm.ainvoke(prompt)
        return str(response.content).strip()

    def _log_query_metrics(
        self,
        question: str,
        sql_query: str,
        execution_time_ms: float,
        row_count: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Structured logging of executed SQL queries and performance metrics."""
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "execution_time_ms": round(execution_time_ms, 2),
            "row_count": row_count,
            "sql_query": sql_query,
            "user_question": question,
            "error": error,
        }
        if status == "success":
            logger.info(f"[QUERY LOG SUCCESS] {log_payload}")
        else:
            logger.error(f"[QUERY LOG ERROR] {log_payload}")

    async def execute_nl_query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        End-to-End natural language database query pipeline:
        1. Introspects database schema dynamically.
        2. Generates SQL via local LLM.
        3. Enforces SELECT-only security guardrails via AST analysis.
        4. Executes query with automatic self-healing retries.
        5. Formats natural language answer.
        6. Logs execution metrics.

        Args:
            question: User natural language inquiry.
            max_retries: Number of retry attempts on syntax execution errors.

        Returns:
            Dictionary containing natural language answer, generated SQL, raw rows, and metrics.
        """
        start_time = time.perf_counter()
        schema_context = self.db.get_schema_context()
        logger.info(f"Processing Text-to-SQL for query: '{question}'")

        # Step 1: Initial SQL Generation
        raw_sql = await self._generate_sql(question, schema_context)

        # Step 2: AST Validation
        validation = SQLValidator.validate(raw_sql)
        if not validation.is_valid:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_query_metrics(question, raw_sql, duration_ms, 0, "validation_rejected", validation.error_message)
            return {
                "status": "error",
                "answer": f"SQL Safety Violation: {validation.error_message}",
                "generated_sql": raw_sql,
                "results": [],
                "execution_time_ms": duration_ms,
            }

        current_sql = validation.cleaned_query

        # Step 3: Execution with Self-Healing Retry Loop
        attempt = 0
        last_error = None
        rows: List[Dict[str, Any]] = []

        while attempt <= max_retries:
            try:
                exec_start = time.perf_counter()
                rows = self.db.execute_query(current_sql)
                exec_duration = (time.perf_counter() - exec_start) * 1000
                total_duration = (time.perf_counter() - start_time) * 1000

                # Query execution succeeded
                self._log_query_metrics(question, current_sql, exec_duration, len(rows), "success")

                # Step 4: Synthesize human-friendly answer
                answer = await self._synthesize_answer(question, rows)

                return {
                    "status": "success",
                    "answer": answer,
                    "generated_sql": current_sql,
                    "results": rows,
                    "execution_time_ms": round(total_duration, 2),
                    "row_count": len(rows),
                    "retry_count": attempt,
                }

            except Exception as db_err:
                attempt += 1
                last_error = str(db_err)
                logger.warning(
                    f"Execution attempt {attempt} failed on query '{current_sql}': {last_error}. "
                    f"Initiating self-healing retry..."
                )

                if attempt <= max_retries:
                    # Self-heal query with error context
                    fixed_raw = await self._fix_sql(question, schema_context, current_sql, last_error)
                    fixed_validation = SQLValidator.validate(fixed_raw)
                    if fixed_validation.is_valid:
                        current_sql = fixed_validation.cleaned_query
                    else:
                        break

        # If all retry attempts failed
        total_duration = (time.perf_counter() - start_time) * 1000
        self._log_query_metrics(question, current_sql, total_duration, 0, "execution_failed", last_error)

        return {
            "status": "error",
            "answer": f"Unable to execute SQL query after {attempt} attempts. Database error: {last_error}",
            "generated_sql": current_sql,
            "results": [],
            "execution_time_ms": round(total_duration, 2),
            "retry_count": attempt,
        }


# Singleton instance of SQLService
sql_service = SQLService()
