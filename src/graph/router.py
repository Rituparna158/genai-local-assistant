"""
Router Module for LangGraph.
Classifies user intent into one of 5 distinct routing pathways:
- 'rag': Company policies, guidelines, IT security, HR rules.
- 'sql': Tabular database questions (orders, customers, products, revenue, stock).
- 'combined': Requires both document policy knowledge and database statistics.
- 'chitchat': Greetings, polite pleasantries, identity questions.
- 'refuse': Out-of-scope, malicious, or prompt injection queries.
"""

import logging
import re
from typing import Literal

from src.api.ollama_service import ollama_service
from src.prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)

RouteType = Literal["rag", "sql", "combined", "chitchat", "refuse"]


class QueryRouter:
    """Classifies user inquiries to route them to the appropriate processing node."""

    def __init__(self):
        self.llm = ollama_service.llm
        self.prompt = ROUTER_PROMPT

    async def classify_route(self, question: str) -> RouteType:
        """
        Classifies user inquiry into 'rag', 'sql', 'combined', 'chitchat', or 'refuse'.
        Includes heuristic safeguards for deterministic greeting, security, and out-of-scope detection.
        """
        cleaned = question.strip().lower()

        # Security edge guardrail: Deterministic check for prompt injections and destructive SQL
        injection_patterns = [
            r"ignore (all )?previous instructions",
            r"bypass (all )?security",
            r"jailbreak",
            r"write (a )?keylogger",
            r"hack (a )?wifi",
            r"^\s*drop\b",
            r"\b(drop\s+table|drop\s+database|truncate\s+table|truncate\s+database)\b",
            r"\b(delete\s+from\s+[a-zA-Z_]+|alter\s+table\s+[a-zA-Z_]+|insert\s+into\s+[a-zA-Z_]+)\b",
        ]
        for pat in injection_patterns:
            if re.search(pat, cleaned):
                logger.warning(
                    f"Malicious/injection query caught by security guardrail: '{question}'"
                )
                return "refuse"

        # Semantic Intent Classification via LLM
        # The LLM evaluates intent dynamically according to decoupled prompts without brittle keyword lists

        # LLM classification
        try:
            formatted = self.prompt.invoke({"question": question})
            try:
                response = await self.llm.ainvoke(formatted)
            except RuntimeError as loop_err:
                if "closed" in str(loop_err).lower():
                    from langchain_ollama import ChatOllama

                    fresh_llm = ChatOllama(
                        model=ollama_service.model_name,
                        base_url=ollama_service.base_url,
                        temperature=0.0,
                    )
                    response = await fresh_llm.ainvoke(formatted)
                else:
                    raise
            raw_route = str(response.content).strip()
            selected_route = self._parse_route_response(raw_route)
            logger.info(f"Router classified question '{question}' -> '{selected_route}'")
            return selected_route
        except Exception as e:
            logger.error(f"Error in router classification: {e}. Defaulting to 'rag'.")
            return "rag"

    def _parse_route_response(self, raw_text: str) -> RouteType:
        """Parses and validates LLM route response with strict token matching and last-verdict precedence."""
        valid_routes: set[str] = {"rag", "sql", "combined", "chitchat", "refuse"}
        cleaned = raw_text.strip().lower().strip("`'\"* .:")

        # 1. Exact match on full response
        if cleaned in valid_routes:
            return cleaned  # type: ignore

        # 2. Check the final line (LLMs often output final verdict on the last line)
        lines = [
            line.strip().strip("`'\"* .:")
            for line in raw_text.strip().lower().splitlines()
            if line.strip()
        ]
        if lines and lines[-1] in valid_routes:
            return lines[-1]  # type: ignore

        # 3. Look for explicit key-value labels: e.g. "route: sql" or "category: rag"
        label_match = re.search(
            r"(?:route|category|classification)\s*[:=-]\s*(rag|sql|combined|chitchat|refuse)\b",
            cleaned,
        )
        if label_match:
            return label_match.group(1)  # type: ignore

        # 4. Find all occurrences; select the last token to respect final conclusions
        matches = re.findall(r"\b(rag|sql|combined|chitchat|refuse)\b", cleaned)
        if matches:
            return matches[-1]  # type: ignore

        logger.warning(
            f"Router received unclassifiable response '{raw_text}'. Defaulting to 'rag'."
        )
        return "rag"


router = QueryRouter()
