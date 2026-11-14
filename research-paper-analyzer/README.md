# Research Paper Analyzer

A class-based, `src/`-layout Python project for research-paper PDF ingestion, retrieval, analysis, and comparison.

## Structure

```text
research-paper-analyzer/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── papers/
├── src/
│   └── research_analyzer/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── analysis.py
│       │   └── comparison.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   └── ingestion.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── chunking.py
│       │   ├── embedding.py
│       │   ├── llm.py
│       │   ├── pdf.py
│       │   └── retrieval.py
│       └── stores/
│           ├── __init__.py
│           └── chroma_store.py
└── tests/
    └── test_pipeline.py
```

## Setup

1. Create and activate a Python 3.10+ virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project:

   ```bash
   pip install -e .
   ```

3. Copy `.env.example` to `.env` and configure the required credentials and local paths. Never commit `.env`, API keys, or certificates.

4. Add selectable PDF papers to `papers/`.

## Commands

The command interface will be implemented in `src/research_analyzer/cli.py`. The intended workflow is:

```bash
research-analyzer ingest ./papers
research-analyzer search "What methodology was used?" --top-k 5
research-analyzer search "What were the results?" --source paper1.pdf
research-analyzer ask "Explain the methodology" --source paper1.pdf
research-analyzer compare "Compare methodologies and limitations" --sources paper1.pdf paper2.pdf
```

## Workflow

- **Ingest:** Read PDFs, split their text into page-aware chunks, generate embeddings, and persist them in the vector store.
- **Search:** Embed a question, retrieve relevant chunks, and retain source/page references.
- **Ask:** Retrieve relevant excerpts and pass them to the analysis agent for a grounded answer.
- **Compare:** Retrieve evidence for each selected paper and pass the excerpts to the comparison agent.

## Design notes and limitations

- The project is organized into services, pipelines, agents, and a persistent store.
- PDF extraction quality depends on the source PDF. Scanned or image-only PDFs may require OCR.
- Retrieval results should be checked against the cited paper and page; generated answers are not a substitute for reading the source.
- Configure model endpoints, credentials, embedding settings, and storage paths through environment variables rather than hard-coding secrets.
