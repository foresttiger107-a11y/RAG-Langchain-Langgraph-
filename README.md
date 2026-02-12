# RAG System with LangChain & LangGraph

A **self-reflective RAG (Retrieval-Augmented Generation)** system built with LangChain and LangGraph. The system decides when to retrieve context, grades document relevance, rewrites questions when needed, and generates answers grounded in retrieved documents.

## Architecture

```
┌─────────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ generate_query_     │────▶│   retrieve   │────▶│ grade_documents │
│ or_respond          │     │   (ToolNode) │     │                 │
└─────────┬───────────┘     └──────────────┘     └────────┬────────┘
          │                                                       │
          │ No tools          yes                     no          │
          ▼                        ┌──────────────────┐           │
       [END]                       │ rewrite_question │◀──────────┘
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ generate_answer │
                                   └────────┬────────┘
                                            │
                                            ▼
                                         [END]
```

### Features

- **Agentic retrieval**: LLM decides when to search the knowledge base vs respond directly
- **Document grading**: Assesses whether retrieved documents are relevant to the question
- **Query rewriting**: Reformulates questions when retrieval returns irrelevant results
- **Vector store**: ChromaDB for semantic search with OpenAI embeddings
- **Multiple document sources**: Load from URLs or local files

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example env file and add your OpenAI API key:

```bash
copy .env.example .env
```

Edit `.env` and set:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Index documents

Index the default blog posts (from Lilian Weng's blog):

```bash
python scripts/index_documents.py
```

Or index only local files in `data/`:

```bash
python scripts/index_documents.py --no-urls --local
```

## Usage

Run the interactive chat:

```bash
python main.py
```

Example questions (after indexing default URLs):

- "What does Lilian Weng say about types of reward hacking?"
- "Hello!" (responds directly without retrieving)
- "Explain diffusion video models"

## Project Structure

```
RAG(Langchain, Langgraph)/
├── main.py              # Entry point, chat interface
├── requirements.txt
├── .env.example
├── data/                # Local documents (optional)
│   └── sample.txt
├── vectorstore/         # ChromaDB persistence (auto-created)
├── scripts/
│   └── index_documents.py
└── src/
    ├── config.py        # Configuration & env vars
    ├── retriever.py     # Document loading, splitting, vector store
    └── graph.py         # LangGraph RAG workflow
```

## Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `LLM_MODEL` | gpt-4o-mini | Model for generation |
| `EMBEDDING_MODEL` | text-embedding-3-small | Model for embeddings |
| `CHUNK_SIZE` | 500 | Document chunk size |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `RETRIEVAL_TOP_K` | 4 | Number of documents to retrieve |

## License

MIT
