"""
Database Connector Module.
Manages SQLAlchemy database connections, connection pooling, ORM sessions,
query execution, and dynamic schema introspection for the SQL Agent.
"""

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from src.config import settings
from src.sql_agent.models import Base

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Manages database connection pool, query execution, and schema introspection."""

    def __init__(self, connection_uri: Optional[str] = None):
        self.uri = connection_uri or settings.db.connection_string
        self.engine: Engine = self._create_resilient_engine()
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def _create_resilient_engine(self) -> Engine:
        """
        Creates an engine connecting to PostgreSQL.
        Falls back to a local SQLite database if PostgreSQL server is unreachable.
        """
        try:
            logger.info(f"Connecting to primary PostgreSQL database at port {settings.db.POSTGRES_PORT}...")
            # pool_pre_ping checks connection health before handing it out
            engine = create_engine(
                self.uri,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={"connect_timeout": 3},
            )
            # Test connectivity
            with engine.connect() as conn:
                conn.execute(text("SELECT 1;"))
            logger.info("Successfully connected to PostgreSQL database.")
            return engine
        except Exception as e:
            fallback_path = Path("data/sql/assistant_fallback.db").resolve()
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Could not connect to PostgreSQL ({e}). "
                f"Falling back to local SQLite engine at: {fallback_path}"
            )
            sqlite_uri = f"sqlite:///{fallback_path}"
            fallback_engine = create_engine(sqlite_uri, pool_pre_ping=True)
            self._initialize_fallback_tables(fallback_engine)
            return fallback_engine

    def _initialize_fallback_tables(self, engine: Engine) -> None:
        """Seeds fallback SQLite database using SQLAlchemy ORM metadata and init.sql."""
        try:
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            if "orders" not in existing_tables:
                logger.info("Creating fallback SQLite tables using ORM metadata...")
                Base.metadata.create_all(bind=engine)

                # Seed initial data from init.sql if available
                init_script_path = Path("data/sql/init.sql")
                if init_script_path.exists():
                    logger.info("Seeding fallback SQLite database with initial records...")
                    raw_sql = init_script_path.read_text(encoding="utf-8")
                    # SQLite doesn't support Postgres-specific sequences or alter commands
                    statements = [
                        s.strip() for s in raw_sql.split(";")
                        if s.strip() and not s.strip().upper().startswith("SELECT SETVAL")
                        and not s.strip().upper().startswith("DROP TABLE")
                        and not s.strip().upper().startswith("CREATE TABLE")
                        and not s.strip().upper().startswith("CREATE INDEX")
                    ]
                    with engine.begin() as conn:
                        for stmt in statements:
                            try:
                                conn.execute(text(stmt))
                            except Exception as seed_err:
                                logger.debug(f"Fallback seed statement skipped: {seed_err}")
        except Exception as err:
            logger.warning(f"Error initializing fallback database: {err}")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager providing a transactional SQLAlchemy ORM session."""
        session: Session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a validated read-only SQL query and returns results as dictionaries.

        Args:
            query: The SQL query string to execute.

        Returns:
            List of row dictionaries mapping column names to values.
        """
        logger.info(f"Executing SQL query: {query.strip()}")
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            keys = result.keys()
            rows = [dict(zip(keys, row)) for row in result.fetchall()]
            logger.info(f"Query returned {len(rows)} row(s).")
            return rows

    def get_schema_context(self) -> str:
        """
        Dynamically extracts table names, columns, data types, and sample data.
        Returns a formatted schema description block for LLM prompt context.
        """
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        schema_lines = []

        for table in table_names:
            columns = inspector.get_columns(table)
            col_descriptions = [f"  - {col['name']} ({col['type']})" for col in columns]
            schema_lines.append(f"Table: {table}\nColumns:\n" + "\n".join(col_descriptions))

            # Fetch up to 2 sample rows to show formatting conventions (e.g. date formats)
            try:
                sample_rows = self.execute_query(f"SELECT * FROM {table} LIMIT 2;")
                if sample_rows:
                    sample_str = str(sample_rows)
                    schema_lines.append(f"Sample data for {table}:\n  {sample_str}")
            except Exception:
                pass
            schema_lines.append("")

        return "\n".join(schema_lines)


# Singleton instance of DatabaseConnector
db_connector = DatabaseConnector()
