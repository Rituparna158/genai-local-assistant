"""
SQL Validator Module.
Strictly validates generated SQL queries using AST token analysis via sqlparse.
Enforces SELECT-only execution and blocks any destructive or multi-statement queries.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DDL, DML, Keyword

logger = logging.getLogger(__name__)


@dataclass
class SQLValidationResult:
    """Data contract representing the validation outcome of a SQL query."""
    is_valid: bool
    cleaned_query: str
    error_message: Optional[str] = None


class SQLValidator:
    """Validates SQL statements against enterprise security guardrails."""

    # Explicit list of forbidden mutating, DDL, and administrative keywords
    FORBIDDEN_KEYWORDS = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "REPLACE", "EXEC", "EXECUTE", "GRANT", "REVOKE",
        "MERGE", "UPSERT", "CALL", "INTO OUTFILE", "DUMPFILE"
    }

    @classmethod
    def clean_query_text(cls, raw_query: str) -> str:
        """Strips markdown code fences, backticks, and extra whitespace from LLM output."""
        cleaned = raw_query.strip()
        # Remove ```sql ... ``` or ``` ... ```
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Strip enclosing single backticks: `SELECT ...`
        if cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()

        # Strip enclosing double/single quotes if the entire statement is wrapped
        if len(cleaned) >= 2:
            if (cleaned[0] == '"' and cleaned[-1] == '"') or (cleaned[0] == "'" and cleaned[-1] == "'"):
                candidate = cleaned[1:-1].strip()
                if candidate.upper().startswith("SELECT") or candidate.upper().startswith("WITH"):
                    cleaned = candidate

        # Clean trailing semicolon
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        return cleaned

    @classmethod
    def validate(cls, raw_query: str) -> SQLValidationResult:
        """
        Validates a SQL query string.
        Ensures the query is:
        1. Non-empty.
        2. A single SQL statement (preventing piggybacked multi-statement injection).
        3. Strictly a SELECT query (or CTE WITH ... SELECT).
        4. Free of any mutating DML/DDL keywords.

        Args:
            raw_query: Raw SQL query text, potentially from an LLM.

        Returns:
            SQLValidationResult indicating validity, cleaned SQL, and error details.
        """
        cleaned = cls.clean_query_text(raw_query)

        if not cleaned:
            return SQLValidationResult(
                is_valid=False,
                cleaned_query="",
                error_message="Query string is empty.",
            )

        # Parse query into AST statements using sqlparse
        parsed_statements = sqlparse.parse(cleaned)

        # 1. Enforce single-statement execution
        if len(parsed_statements) == 0:
            return SQLValidationResult(
                is_valid=False,
                cleaned_query=cleaned,
                error_message="Could not parse valid SQL statement.",
            )

        if len(parsed_statements) > 1:
            logger.warning(f"Multiple SQL statements detected and rejected: {cleaned}")
            return SQLValidationResult(
                is_valid=False,
                cleaned_query=cleaned,
                error_message="Multi-statement queries are strictly forbidden. Only a single query is allowed.",
            )

        statement: Statement = parsed_statements[0]

        # 2. Inspect root command type
        stmt_type = statement.get_type()
        if stmt_type != "SELECT" and not cleaned.upper().startswith("WITH"):
            logger.warning(f"Non-SELECT statement type '{stmt_type}' rejected: {cleaned}")
            return SQLValidationResult(
                is_valid=False,
                cleaned_query=cleaned,
                error_message=f"Only SELECT queries are permitted. Detected statement type: '{stmt_type}'.",
            )

        # 3. Token-level keyword scanning for forbidden tokens
        flattened_tokens = [
            token.value.upper()
            for token in statement.flatten()
            if not token.is_whitespace
        ]

        for word in flattened_tokens:
            if word in cls.FORBIDDEN_KEYWORDS:
                logger.warning(f"Forbidden keyword '{word}' detected and blocked in SQL: {cleaned}")
                return SQLValidationResult(
                    is_valid=False,
                    cleaned_query=cleaned,
                    error_message=f"Forbidden keyword '{word}' detected. Mutation and DDL operations are blocked.",
                )

        # Clean trailing semicolon for standardized execution
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        return SQLValidationResult(
            is_valid=True,
            cleaned_query=cleaned,
            error_message=None,
        )
