# Anagnosis

Anagnosis helps students and researchers turn PDFs, lecture notes, and messy documents into searchable, citation-ready knowledge. It surfaces definitions, formulas, and examples so you spend less time skimming and more time understanding.

Overview

- Search by concept (not just keywords).
- Answers with page references for easy citation.
- Better handling of math, tables and structured academic content.

Demo

![Demo thumbnail](https://github.com/user-attachments/assets/d8cc2b64-f69a-4974-8e57-8a0edc442cea)

Full demo video:

https://github.com/user-attachments/assets/d8cc2b64-f69a-4974-8e57-8a0edc442cea

Quick start (local)

1. Install dependencies:

	pip install -r requirements.txt

2. Run the app (single-process dev):

	uvicorn serve:app --host 0.0.0.0 --port 8080

3. Health check:

	curl http://127.0.0.1:8000/healthz

Where to look

- `serve.py` — ASGI entrypoint for API + UI.
- `api/` — backend endpoints and services.
- `app/` — desktop GUI and helpers.

Status

Experimental — the repo contains working parsing, retrieval and UI components and is being actively developed.

Contributing & contact

Contributions welcome. Open issues or pull requests, or check the repository docs for more details.
