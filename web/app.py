"""Gradio-powered web interface for the Anagnosis pipeline."""
from __future__ import annotations

import os
import pathlib
import queue
import sys
import threading
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import pipeline
from api.services.pipeline import PipelineCancelled


def _format_log_lines(lines: Sequence[str], title: str) -> str:
    if not lines:
        return f"### {title}\n- (idle)"
    body = "\n".join(f"- {line}" for line in lines)
    return f"### {title}\n{body}"


def _format_ingest_details(details: Sequence[Dict[str, Any]]) -> str:
    if not details:
        return "No documents ingested yet. Upload PDFs to build your library."
    lines = ["| File | Pages | OCR Pages | Chunks | Suspect Pages |", "| --- | --- | --- | --- | --- |"]
    for row in details:
        name = pathlib.Path(row.get("file", "")).name or "?"
        num_pages = row.get("num_pages", "?")
        num_chunks = row.get("num_chunks", "?")
        ocr_pages = row.get("ocr_page_numbers") or []
        suspect_pages = row.get("suspect_pages") or []
        ocr_preview = ", ".join(str(x) for x in ocr_pages[:6]) or "—"
        if len(ocr_pages) > 6:
            ocr_preview += " …"
        suspect_preview = ", ".join(str(x) for x in suspect_pages[:6]) or "—"
        if len(suspect_pages) > 6:
            suspect_preview += " …"
        lines.append(
            f"| {name} | {num_pages} | {ocr_preview} | {num_chunks} | {suspect_preview} |"
        )
    return "\n".join(lines)


def _format_quotes(quotes: Sequence[Dict[str, Any]]) -> str:
    if not quotes:
        return ""
    blocks: List[str] = ["", "### Evidence snippets"]
    for q in quotes:
        quote = q.get("quote") or q.get("text") or ""
        source = q.get("source") or q.get("citation") or ""
        if not quote:
            continue
        block = f"> {quote}\n>\n> — {source}" if source else f"> {quote}"
        blocks.append(block)
    return "\n".join(blocks)


def _format_agent_diag(result: Dict[str, Any], agents_enabled: bool) -> str:
    meta = result.get("agent_meta") if isinstance(result, dict) else None
    report = result.get("agent_report") if isinstance(result, dict) else None
    if meta and meta.get("enabled"):
        verdict = meta.get("verdict", "ran")
        if meta.get("changed"):
            verdict = "modified"
        summary_bits: List[str] = []
        kept = meta.get("kept_sentences")
        total = meta.get("total_sentences")
        if kept is not None and total:
            summary_bits.append(f"{kept}/{total} sentences kept")
        status_counts = meta.get("status_counts") or {}
        if status_counts.get("supported"):
            summary_bits.append(f"{status_counts['supported']} supported")
        if status_counts.get("weak"):
            summary_bits.append(f"{status_counts['weak']} need review")
        if meta.get("time_sec") is not None:
            summary_bits.append(f"{meta['time_sec']:.2f}s")
        summary = ", ".join(summary_bits) or "completed"
        diag = f"\n\n_Agents ({verdict}): {summary}._"
        if isinstance(report, str) and report.strip():
            diag += "\n\n<details><summary>Agent report</summary>\n\n" + report + "\n\n</details>"
        web_rep = meta.get("web_results_md")
        if isinstance(web_rep, str) and web_rep.strip():
            diag += "\n\n<details><summary>Web evidence</summary>\n\n" + web_rep + "\n\n</details>"
        return diag
    if agents_enabled:
        return "\n\n_Agents: ran._"
    return ""


def _format_answer(question: str, result: Dict[str, Any], agents_enabled: bool) -> str:
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    citations = result.get("citations") if isinstance(result, dict) else []
    quotes = result.get("quotes") if isinstance(result, dict) else []
    cite_block = f"\n\n**Citations:** {', '.join(citations)}" if citations else ""
    quote_block = _format_quotes(quotes)
    agent_block = _format_agent_diag(result if isinstance(result, dict) else {}, agents_enabled)
    header = f"### Q: {question}\n\n" if question else ""
    return header + answer + cite_block + quote_block + agent_block


def _ingest_action(files: List[str]):
    title = "Ingestion log"
    if not files:
        yield (
            "Upload one or more PDFs to build your knowledge base.",
            "",
            _format_log_lines([], title),
        )
        return
    paths = [pathlib.Path(f) for f in files if f]
    if not paths:
        yield (
            "No readable files provided.",
            "",
            _format_log_lines([], title),
        )
        return

    logs: List[str] = []
    q_logs: "queue.Queue[str]" = queue.Queue()
    result_box: Dict[str, Any] = {}
    done = threading.Event()

    def log(msg: str) -> None:
        logs.append(msg)
        try:
            print(f"[ingest] {msg}", flush=True)
        except Exception:
            pass
        q_logs.put(_format_log_lines(logs, title))

    def worker() -> None:
        try:
            result_box["result"] = pipeline.ingest_documents(paths, progress=log)
        except PipelineCancelled:
            result_box["error"] = ("cancelled", "Ingestion cancelled.")
        except Exception as exc:
            result_box["error"] = ("error", exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    summary_md = "Starting ingestion…"
    detail_md = ""
    current_logs = _format_log_lines(["Queued for ingestion"], title)
    yield summary_md, detail_md, current_logs

    while not done.is_set() or not q_logs.empty():
        try:
            current_logs = q_logs.get(timeout=0.2)
            yield summary_md, detail_md, current_logs
        except queue.Empty:
            pass

    thread.join()

    if "error" in result_box:
        kind, info = result_box["error"]
        msg = info if isinstance(info, str) else str(info)
        if kind == "cancelled":
            logs.append("Ingestion cancelled.")
            final = _format_log_lines(logs, title)
            yield "Ingestion cancelled.", detail_md, final
            return
        err_block = f"```\\n{msg}\\n```"
        logs.append(str(msg))
        final = _format_log_lines(logs, title)
        yield err_block, detail_md, final
        return

    result = result_box.get("result", {})
    summary_md = result.get("doc_summary") or "No documents ingested."
    detail_md = _format_ingest_details(result.get("details", []))
    logs.append("Ingestion complete.")
    log_md = _format_log_lines(logs, title)
    yield summary_md, detail_md, log_md



def _question_action(
    question: str,
    chat_history: List[Tuple[str, str]],
    stored_history: List[Dict[str, str]],
    formula_mode: bool,
    agents_enabled: bool,
    web_enabled: bool,
):
    title = "Ask log"
    if not question.strip():
        yield question, chat_history, stored_history, _format_log_lines(["Please enter a question."], title)
        return

    logs: List[str] = []
    q_logs: "queue.Queue[str]" = queue.Queue()
    result_box: Dict[str, Any] = {}
    done = threading.Event()

    def log(msg: str) -> None:
        logs.append(msg)
        try:
            print(f"[ask] {msg}", flush=True)
        except Exception:
            pass
        q_logs.put(_format_log_lines(logs, title))

    def worker() -> None:
        try:
            result_box["result"] = pipeline.answer_question(
                question,
                history=stored_history,
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                progress=log,
            )
        except PipelineCancelled:
            result_box["error"] = ("cancelled", "Answering cancelled.")
        except Exception as exc:
            result_box["error"] = ("error", exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    current_logs = _format_log_lines(["Starting retrieval…"], title)
    yield question, chat_history, stored_history, current_logs

    while not done.is_set() or not q_logs.empty():
        try:
            current_logs = q_logs.get(timeout=0.2)
            yield question, chat_history, stored_history, current_logs
        except queue.Empty:
            pass

    thread.join()

    if "error" in result_box:
        kind, info = result_box["error"]
        logs.append(str(info))
        log_text = current_logs or _format_log_lines(logs, title)
        if kind == "cancelled":
            yield "", chat_history, stored_history, log_text
            return
        new_chat = chat_history + [(question, f"**Error:** {info}")]
        yield "", new_chat, stored_history, log_text
        return

    result = result_box.get("result", {})
    formatted = _format_answer(question, result, agents_enabled)
    new_chat = chat_history + [(question, formatted)]
    new_history = stored_history + [{"q": question, "a": formatted}]
    logs.append("Answer ready.")
    log_text = _format_log_lines(logs, title)
    yield "", new_chat, new_history, log_text



def _clear_conversation() -> Tuple[List[Tuple[str, str]], List[Dict[str, str]], str]:
    return [], [], _format_log_lines(["Conversation cleared."], "Ask log")


def build_demo() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="Anagnosis Web") as demo:
        gr.Markdown(
            """
            # 📚 Anagnosis — Web Interface
            Bring your RAG pipeline to the browser: upload PDFs, build semantic memory, and ask rich questions with citations and formula support.
            """
        )
        with gr.Tab("Library"):
            file_input = gr.File(
                label="Upload PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
            )
            ingest_btn = gr.Button("Ingest Documents", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    ingest_summary = gr.Markdown(label="Document summaries", value="_No documents ingested yet._")
                with gr.Column(scale=2):
                    ingest_details = gr.Markdown(label="Ingestion details", value="")
            with gr.Accordion("Pipeline log", open=False):
                ingest_logs = gr.Markdown(value=_format_log_lines([], "Ingestion log"))

            ingest_btn.click(
                _ingest_action,
                inputs=file_input,
                outputs=[ingest_summary, ingest_details, ingest_logs],
            )

        with gr.Tab("Ask"):
            chatbot = gr.Chatbot(label="Conversation", height=420)
            history_state = gr.State([])  # list of {q, a}
            question_box = gr.Textbox(
                label="Ask a question",
                placeholder="e.g., Summarize the main theorems from Chapter 3",
                lines=2,
            )
            with gr.Row():
                formula_cb = gr.Checkbox(label="Formula mode", value=False, info="Extract canonical formulas with LaTeX output")
                agents_cb = gr.Checkbox(label="Agents", value=False, info="Enable agentic verification (slower)")
                web_cb = gr.Checkbox(label="Web search", value=False, info="Blend local retrieval with web results")
            ask_btn = gr.Button("Ask", variant="primary")
            reset_btn = gr.Button("Clear conversation")
            with gr.Accordion("Pipeline log", open=False):
                ask_logs = gr.Markdown(value=_format_log_lines([], "Ask log"))

            ask_btn.click(
                _question_action,
                inputs=[question_box, chatbot, history_state, formula_cb, agents_cb, web_cb],
                outputs=[question_box, chatbot, history_state, ask_logs],
            )
            reset_btn.click(
                _clear_conversation,
                inputs=None,
                outputs=[chatbot, history_state, ask_logs],
                queue=False,
            )

    return demo



DEMO = build_demo()


def create_app() -> FastAPI:
    app = FastAPI(title="Anagnosis Web Service")
    gr.mount_gradio_app(app, DEMO, path="/ui")

    @app.get("/")
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
