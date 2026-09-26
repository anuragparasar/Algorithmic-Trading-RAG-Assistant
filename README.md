# Algorithmic Trading RAG Assistant

## Overview
The Algorithmic Trading RAG Assistant is a specialized Retrieval-Augmented Generation (RAG) pipeline designed to ingest, index, and query technical trading documentation, API schemas, and synthetic index specifications for the Deriv platform[cite: 1, 2]. It uses a local hybrid search system and strict deterministic guardrails to provide highly accurate, hallucination-free technical specifications to algorithmic traders and developers[cite: 2].

## System Architecture
* **Resilient Data Ingestion:** The pipeline fetches live JSON schemas and Markdown documentation, utilizing a 2-attempt HTTP retry loop with hardcoded fallbacks to handle network anomalies[cite: 1]. It features a header-aware Markdown chunker and a custom regex tokenizer that preserves alphanumeric financial terms (e.g., `R_100`, `stop_loss`)[cite: 1].
* **Hybrid Retrieval (RRF):** Queries are processed through a dual-index system combining dense vector search (`all-MiniLM-L6-v2`) stored in Qdrant and sparse keyword matching (BM25)[cite: 1]. Results are merged using Reciprocal Rank Fusion (k=60)[cite: 1, 2].
* **Cross-Encoder Reranking:** The top 10 candidates from the fused retrieval list are reranked using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to surface the most contextually relevant chunks[cite: 2].
* **LLM Generation & Guardrails:** Context is passed to OpenAI's `gpt-4o-mini` to generate structured Pydantic outputs[cite: 2]. A deterministic regex guardrail then scans the generated text and extracted parameters, validating them against the retrieved source chunks using word-boundaries to block numerical hallucinations[cite: 2].

## Installation
1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies (inferred from the project files):
   ```bash
   pip install qdrant-client sentence-transformers rank-bm25 openai pydantic requests beautifulsoup4 numpy
   ```
4. Set your OpenAI API key as an environment variable[cite: 2]:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage
The system is divided into two primary execution phases: data ingestion and querying.

**1. Build the Vector and Sparse Indexes**
Run the ingestion pipeline to fetch live data, process local specifications, generate embeddings, and build the persistent Qdrant database and BM25 index[cite: 1].
```bash
python deriv_rag_pipeline.py
```

**2. Query the Trading Assistant**
Once the database is populated, execute the assistant script to run hybrid queries against the indexed knowledge base. The system will retrieve context, generate a structured answer, run the guardrail checks, and output the exact sources for each extracted parameter[cite: 2].
```bash
python deriv_trading_assistant.py
```
