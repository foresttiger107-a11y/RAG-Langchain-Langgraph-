"""Main entry point for the RAG system."""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OPENAI_API_KEY, VECTORSTORE_DIR
from src.graph import build_rag_graph
from src.retriever import get_retriever, load_existing_vectorstore


def main():
    """Run the RAG chat interface."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is required. Copy .env.example to .env and set your key.")
        sys.exit(1)

    if not VECTORSTORE_DIR.exists():
        print(
            "Error: Vector store not found. Run indexing first:\n"
            "  python scripts/index_documents.py"
        )
        sys.exit(1)

    print("Loading RAG system...")
    vectorstore = load_existing_vectorstore(persist_directory=VECTORSTORE_DIR)
    retriever = get_retriever(vectorstore)
    graph = build_rag_graph(retriever)

    print("\nRAG System Ready! Ask questions about the indexed documents.")
    print("Commands: /quit to exit\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))
        inputs = {"messages": messages}

        try:
            result = graph.invoke(inputs)
            messages = result.get("messages", messages)
            # Find the last AI message with content
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "ai" and getattr(msg, "content", ""):
                    print(f"\nAssistant: {msg.content}\n")
                    break
        except Exception as e:
            print(f"\nError: {e}\n")
            messages.pop()


if __name__ == "__main__":
    main()
