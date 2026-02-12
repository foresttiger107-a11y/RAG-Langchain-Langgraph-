"""LangGraph RAG workflow with self-reflective retrieval."""

from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .config import LLM_MODEL, OPENAI_API_KEY


# Prompts
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n-------\n{question}\n-------\n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base. "
    "When users ask questions that might be answered from the knowledge base, use the retriever tool to search for relevant information. "
    "For casual greetings or questions outside the knowledge domain, respond directly without retrieving."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def build_rag_graph(retriever):
    """Build the agentic RAG graph with retrieval, grading, and answer generation."""

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")

    response_model = ChatOpenAI(model=LLM_MODEL, temperature=0)
    grader_model = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # --- Retriever Tool ---
    @tool
    def retrieve_documents(query: str) -> str:
        """Search and return relevant information from the knowledge base."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = retrieve_documents

    # --- Nodes ---
    def generate_query_or_respond(state):
        """Call the model to decide: retrieve or respond directly."""
        messages = state["messages"]
        # Prepend system prompt when not present
        if not any(getattr(m, "type", None) == "system" for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = response_model.bind_tools([retriever_tool]).invoke(messages)
        return {"messages": [response]}

    def _get_user_question(messages):
        """Extract the first user question from messages."""
        for m in messages:
            if getattr(m, "type", None) == "human":
                return m.content if hasattr(m, "content") else str(m)
        return messages[0].content if messages and hasattr(messages[0], "content") else ""

    def grade_documents(state) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant."""
        messages = state["messages"]
        question = _get_user_question(messages)
        context = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def rewrite_question(state):
        """Rewrite the original user question for better retrieval."""
        messages = state["messages"]
        question = _get_user_question(messages)

        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(state):
        """Generate final answer from question and retrieved context."""
        messages = state["messages"]
        question = _get_user_question(messages)
        context = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    # --- Graph Assembly ---
    workflow = StateGraph(MessagesState)

    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", "__end__": END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()
