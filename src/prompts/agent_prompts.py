"""
Agent Prompts Module.
Defines reusable prompt templates for LangGraph agent nodes:
contextualization, multi-domain synthesis, conversational chitchat, and guardrail refusals.
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt for query contextualization and resolving anaphoric references
CONTEXTUALIZE_SYSTEM_TEMPLATE = (
    "You are a conversational query context resolver.\n"
    "Given the chat history and the user's latest message which may contain follow-up references "
    "or pronouns (e.g., 'it', 'they', 'that product', 'its price', 'their email', 'that policy'), "
    "formulate a standalone question that can be understood without the chat history.\n\n"
    "RULES:\n"
    "1. If the latest message is already a standalone question or a greeting, return it unchanged.\n"
    "2. Do NOT answer the question. Only reformulate it to resolve references from previous turns.\n"
    "3. Return ONLY the reformulated question text without quotes, explanation, or markdown."
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_TEMPLATE),
        ("human", "CHAT HISTORY:\n{chat_history}\n\nLATEST QUESTION: {question}"),
    ]
)

# System prompt for synthesizing combined policy documents (RAG) and relational records (SQL)
COMBINED_SYNTHESIS_SYSTEM_TEMPLATE = (
    "You are an enterprise data assistant.\n"
    "The user has asked a question that requires BOTH company policy context and database records.\n\n"
    "COMPANY POLICY FINDINGS:\n{rag_answer}\n\n"
    "DATABASE ANALYSIS:\n{sql_answer}\n(SQL Query: {sql_query})\n\n"
    "STRICT GROUNDING RULES:\n"
    "1. Extract company policy rules, windows, and guidelines EXCLUSIVELY from COMPANY POLICY FINDINGS.\n"
    "   - Present the policy facts accurately (such as return windows, core hours, leave days, or security rules).\n"
    "   - Ignore any disclaimers in COMPANY POLICY FINDINGS regarding missing database statistics or order records.\n"
    "2. Extract numbers, counts, and financial metrics EXCLUSIVELY from DATABASE ANALYSIS.\n"
    "   - Ignore any SQL disclaimers, 'N/A', or NULL values concerning policy terms.\n"
    "3. Present both the verified policy rules and database metrics together in a clear, well-structured response.\n"
    "4. Under no circumstances should you invent, assume, or fabricate external facts.\n"
    "5. Be factual, professional, and concise."
)

COMBINED_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", COMBINED_SYNTHESIS_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)

# System prompt for greetings and polite pleasantries
CHITCHAT_SYSTEM_TEMPLATE = (
    "You are the Enterprise Data Assistant.\n"
    "You assist employees with internal company policies (leave, benefits, IT security, remote work) "
    "and business database queries (customers, products, orders, sales metrics).\n"
    "Introduce yourself clearly as 'Enterprise Data Assistant' (never use placeholders like [Assistant Name]).\n"
    "Respond to the user's greeting warmly, concisely, and offer your help with company documents or data."
)

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CHITCHAT_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)

# Standard refusal response when a request violates safety rules or is out-of-scope
GUARDRAIL_REFUSAL_MESSAGE = (
    "I am an enterprise data assistant designed to assist with internal company policy documents "
    "and relational database queries (customers, products, and orders). "
    "Destructive database operations (such as DROP, TRUNCATE, DELETE, or ALTER) and requests "
    "outside this operational scope cannot be processed."
)

# Default conversational prompt for direct LLM interactions
DEFAULT_CHAT_SYSTEM_TEMPLATE = (
    "You are a helpful, accurate, and concise AI assistant. "
    "Provide direct and factual answers to user questions."
)
DEFAULT_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEFAULT_CHAT_SYSTEM_TEMPLATE),
        ("human", "{user_message}"),
    ]
)
