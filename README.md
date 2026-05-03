# Amazon RAG: Multi-Agent Recommendation System

A Retrieval-Augmented Generation (RAG) system built to explore the integration of AI agents with the Amazon Product Reviews dataset.

This project serves as a practical exploration of LangGraph, pgvector, and Hybrid Search, combining Semantic and Full-Text Search for improved retrieval.

## How It Works (The Multi-Agent Architecture)

When a user submits a query, it is processed by a coordinated pipeline of four specialized agents:

1. **Orchestrator**: Acts as the router. It analyzes the user's intent to determine if they are initiating a conversational request or actively searching for products.
2. **Finder**: The core search agent. It queries PostgreSQL using Hybrid Search (Reciprocal Rank Fusion) to retrieve the most relevant product reviews based on the user's input.
3. **Recommender**: The cross-selling agent. It analyzes the purchase history of users who interacted with the items found by the Finder, returning "frequently bought together" recommendations.
4. **Synthesizer**: The response generator. It aggregates the raw data retrieved by the Finder and Recommender, formatting it into a natural and helpful response for the user.

## The Tech Stack

- **Orchestration**: [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) (Stateful Multi-Agent Workflows)
- **LLMs & Embeddings**: OpenAI (`gpt-4o`, `text-embedding-3-small`)
- **Database**: PostgreSQL with the `pgvector` extension
- **Search Implementation**: Reciprocal Rank Fusion (RRF) combining cosine distance and PostgreSQL `tsvector` full-text search.
- **Package Manager**: `uv`

## Getting Started

### 1. Prerequisites

Ensure you have the following installed and configured:

- Docker & Docker Compose
- Python 3.12+ and `uv`
- An active OpenAI API Key
- Kaggle

### 2. Setup

Clone the repository and create your `.env` file in the root directory to define your environment variables:

```env
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBEDDING_MODEL=
OPENAI_JUDGE_MODEL=
```

Install the dependencies using `uv`:

```bash
uv sync
```

### 3. Start the Database

Spin up the PostgreSQL and pgAdmin containers:

```bash
docker compose up -d
```

### 4. Data Pipeline

To populate the database with Amazon reviews, run the following scripts sequentially. These scripts are designed to be resumable in case of interruption:

```bash
# 1. Download & clean the Kaggle dataset, then load it into PostgreSQL
uv run python utility/ingest_amazon_reviews.py

# 2. Generate OpenAI Embeddings (Vectors)
uv run python utility/generate_embeddings.py

# 3. Create the Full-Text Search index
uv run python -m utility.migrate_fts
```

### 5. Running the Application

Launch the main application to interact with the agents. The system includes a basic authentication flow to maintain user context.

```bash
uv run python main.py
```

## Evaluation Pipeline

The project includes an evaluation suite to compare the effectiveness of Hybrid Search against standard Semantic Search:

1. **Generate Gold Truth**: Uses an LLM to build a test dataset of realistic user queries grounded in actual product data.
2. **Evaluate**: Tests Semantic Search vs. Hybrid Search (RRF) and generates a report comparing Hit Rate and Mean Reciprocal Rank (MRR).

```bash
uv run python -m evaluation.generate_gold_truth
uv run python -m evaluation.evaluate
```

## Future Ideas

Since this is an experimental project, there are several areas for potential expansion:

- **Product-Level Summaries**: Use an LLM to aggregate and summarize individual reviews into a clean, structured product catalog.
- **Order Management & Support Agent**: Generate synthetic e-commerce order data (tracking numbers, shipping statuses, return windows) and introduce a dedicated support agent capable of securely handling post-purchase inquiries alongside recommendations.
