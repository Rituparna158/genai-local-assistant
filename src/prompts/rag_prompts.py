"""
RAG Prompts Module.
Defines reusable prompt templates for document-grounded question answering and citation synthesis.
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt strictly enforcing context grounding, synonyms matching, and absence handling
RAG_SYSTEM_TEMPLATE = (
    "You are an accurate enterprise data assistant.\n"
    "Answer the user's question clearly and factually using the information provided in the CONTEXT below.\n"
    "Match equivalent terms and synonyms appropriately (e.g., 'carry forward' corresponds to 'carry-over' or 'carried into', 'vacation' corresponds to 'annual leave').\n"
    "Cite specific numbers, days, and rules mentioned in the context.\n"
    "If the topic is genuinely not addressed anywhere in the CONTEXT, respond with: "
    "'I cannot find information regarding this in the company documents.'\n\n"
    "CONTEXT:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)
