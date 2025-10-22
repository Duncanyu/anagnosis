# Anagnosis  

**AI-powered document intelligence for university students and researchers**  

Anagnosis is a project designed to help students, researchers, and academics work more effectively with complex documents. Instead of skimming hundreds of pages or losing track of formulas buried in PDFs, Anagnosis transforms unstructured academic files into searchable, citation-ready knowledge.  

Built on top of a custom **retrieval-augmented generation (RAG)** pipeline, it integrates **multi-strategy document parsing, AI-driven formula recognition, semantic retrieval, and conversation memory** into a single workflow. The goal is simple: reduce time wasted searching and increase time spent understanding.  

> ⚠️ **Status:** Work in Progress — this is not a finished tool. It’s an experimental project that also serves as a demonstration of applied skills in RAG, LLM orchestration, reranking, ONNX deployment, and system design.  

---

## Project Objectives  

- Make dense academic PDFs easier to navigate.  
- Provide **citation-ready answers** backed by page references.  
- Build automatic **formula sheets** by extracting mathematical content.  
- Enable **semantic search** across lecture notes, research papers, and textbooks.  
- Create a **persistent conversation layer** so follow-up questions keep context.  
- Deliver knowledge through both a **desktop GUI** and an **API server**.  

---

## Key Features  

- **Document Parsing**: PyMuPDF, PDFMiner, Poppler, and OCR fallback with glyph recovery.  
- **Formula Intelligence**: ONNX-based MiniLM classifier, formula mode, canonical LaTeX export.  
- **Semantic Retrieval**: FAISS embeddings, MMR ordering, SFT reranking, cross-encoder integration.  
- **Chunking & Embeddings**: Structure-preserving segmentation, math/table detection, token overlap control.  
- **Conversation Memory**: Persistent history with intelligent pruning.  
- **APIs & Interfaces**: REST endpoints (FastAPI) and PySide6 GUI.  

---

## System Overview  

<img width="1147" height="468" alt="Screenshot 2025-09-25 at 11 06 08" src="https://github.com/user-attachments/assets/12f83b71-0489-4de3-9de5-45a73e7ba41e" />

---

## Tech Stack  

- **RAG pipeline** with FAISS + MMR + cross-encoder reranking  
- **LLMs**: OpenAI GPT, HuggingFace backends, vLLM server support  
- **SFT reranking**: Supervised fine-tuned relevance models  
- **Formula classification**: ONNX Runtime (MiniLM custom model)  
- **OCR**: Tesseract-based fallback with math-aware configs  
- **Embeddings**: HuggingFace SentenceTransformers  
- **Interfaces**: FastAPI REST API + PySide6 GUI  

---

## Current Status  

- Document parsing pipeline in place with OCR fallbacks  
- Formula detection and classification working with ONNX model  
- Semantic retrieval with FAISS index + reranking enabled  
- Basic GUI and REST API functional  
- Citation-ready outputs available  
- Conversation memory system integrated  

---

## Why This Matters  

University students and researchers often face the same problem: too much material, too little time. Traditional search is keyword-based and fails on PDFs filled with equations, tables, and diagrams. Anagnosis shows how modern **AI, embeddings, and retrieval techniques** can be used to bridge that gap — surfacing formulas, definitions, and explanations directly when needed.  

This project is both a **practical tool-in-progress** for academic life and a **demonstration of technical skills** in advanced AI system design.  

---

## Quickstart (Local, no Docker)

- Install dependencies: `python3 -m pip install -r requirements.txt`
- Start API (SQLite by default): `bash scripts/run_local_api.sh`
  - Health check: `curl http://127.0.0.1:8000/healthz`
- Start Web UI (separate terminal): `bash scripts/run_local_web.sh`
  - Open `http://127.0.0.1:7860`

Notes
- These scripts avoid Docker Compose and default to SQLite (`./anagnosis.db`).
- To use Postgres locally instead, run the API with `USE_SQLITE=0` and set `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@localhost:5432/db`).
- The `.env` file is intended for Docker Compose. It is not automatically loaded for local runs.

---

## Single Command Deploy (Render)

This repository now includes a unified ASGI entrypoint that serves both the API and the Web UI from a single process.

- Entry file: `serve.py`
- ASGI app: `serve:app`

On Render, create a Web Service pointing at this repo and set:

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn serve:app --host 0.0.0.0 --port $PORT`

Environment
- Optional: set `DATABASE_URL` for Postgres; otherwise it defaults to a local SQLite file (`./anagnosis.db`).
- Optional: set `OPENAI_API_KEY`, `HF_TOKEN`, etc., per your configuration needs.

Local single-process run
- Use a port other than `7860` so the frontend uses same-origin `/api`:
  - `uvicorn serve:app --host 0.0.0.0 --port 8080`
  - Open `http://localhost:8080`
- If you use port `7860`, the frontend will try to call an API on `:8000` (intended for two-process dev). Either run the API separately on 8000 or choose a different port as above.

If you see a Postgres connection error locally
- Your shell may have `DATABASE_URL` set to a Docker-only hostname like `postgres` from a previous `docker-compose` run.
- Quick fix: force SQLite for local runs by prefixing the command:
  - `FORCE_SQLITE=1 uvicorn serve:app --host 0.0.0.0 --port 8080`
- Or unset `DATABASE_URL` in your shell so the app falls back to SQLite (`sqlite:///./anagnosis.db`).
