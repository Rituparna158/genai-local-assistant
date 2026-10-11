"""
Unit and integration tests for the SQL Service and Database Connector.
Validates execution pipeline, metrics logging, safety guards, and schema introspection.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.sql_agent.db_connector import db_connector
from src.sql_agent.sql_service import SQLService


class TestDatabaseConnector:
    """Validates DatabaseConnector functionality."""

    def test_schema_context_contains_all_core_tables(self):
        """Ensures schema introspection retrieves customers, products, and orders."""
        schema_text = db_connector.get_schema_context()
        assert "Table: customers" in schema_text or "customers" in schema_text.lower()
        assert "Table: products" in schema_text or "products" in schema_text.lower()
        assert "Table: orders" in schema_text or "orders" in schema_text.lower()

    def test_execute_query_returns_row_dictionaries(self):
        """Ensures execute_query returns valid row dictionaries."""
        rows = db_connector.execute_query("SELECT id, name FROM customers LIMIT 2;")
        assert isinstance(rows, list)
        if len(rows) > 0:
            assert "id" in rows[0]
            assert "name" in rows[0]


class TestSQLService:
    """Validates the full SQLService pipeline and guardrails."""

    def test_execute_nl_query_rejects_unsafe_sql(self):
        """Verifies that an attempt to generate mutating SQL is blocked by the validator."""
        service = SQLService()
        # Mock LLM to return a malicious DROP TABLE statement
        service.llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "DROP TABLE customers;"
        service.llm.ainvoke.return_value = mock_response

        result = asyncio.run(service.execute_nl_query("Delete all customer records"))
        assert result["status"] == "error"
        assert "SQL Safety Violation" in result["answer"]
        assert len(result["results"]) == 0

    def test_execute_nl_query_success_flow(self):
        """Verifies successful end-to-end Text-to-SQL flow with valid SELECT."""
        service = SQLService()

        # Mock LLM generation and answer synthesis
        service._generate_sql = AsyncMock(
            return_value="SELECT name, country FROM customers LIMIT 2;"
        )
        service._synthesize_answer = AsyncMock(
            return_value="The first two customers are TechCorp Solutions in United States and Acme Logistics in Canada."
        )

        result = asyncio.run(service.execute_nl_query("List 2 customers and their countries"))

        assert result["status"] == "success"
        assert "TechCorp Solutions" in result["answer"]
        assert result["generated_sql"] == "SELECT name, country FROM customers LIMIT 2"
        assert result["row_count"] >= 1
        assert "execution_time_ms" in result

    def test_execute_nl_query_self_healing_retry(self):
        """Verifies self-healing retry triggers when initial SQL has a syntax error."""
        service = SQLService()

        # First query has a bad table name, fix provides the right table name
        service._generate_sql = AsyncMock(return_value="SELECT * FROM non_existent_table_xyz;")
        service._fix_sql = AsyncMock(return_value="SELECT * FROM customers LIMIT 1;")
        service._synthesize_answer = AsyncMock(return_value="Found 1 customer.")

        result = asyncio.run(service.execute_nl_query("Show me a customer"))

        assert result["status"] == "success"
        assert result["retry_count"] == 1
        assert "customers" in result["generated_sql"]
