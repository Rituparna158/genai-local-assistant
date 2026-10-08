"""
Frontend Configuration Module.
Defines API endpoints, default settings, and sample demo queries.
"""

import os
from typing import Dict, List

# FastAPI Backend Base URL
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

# Predefined prompt chips for 1-click testing across all 4 routing modes
SAMPLE_QUERIES: Dict[str, List[str]] = {
    "📊 SQL Analytics": [
        "Who are our top 3 customers by spending?",
        "Which product has the highest stock quantity?",
        "What is our total revenue from completed orders?",
    ],
    "📄 Document RAG": [
        "What is our annual leave entitlement according to the policy?",
        "What are the remote work security guidelines?",
        "What is the company probation period policy?",
    ],
    "🔀 Combined (RAG + SQL)": [
        "What is our return window in the refund policy, and how many orders were refunded in September?",
    ],
    "💬 Greetings": [
        "Hello! Who are you and how can you help me?",
    ],
}
