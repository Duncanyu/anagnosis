"""Shared ingestion and question-answering pipeline utilities."""
from __future__ import annotations

import os
import pathlib
import re
import json
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

from api.services.parse import parse_any_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks, search
from api.services.summarize import (
    summarize_document,
    summarize_all_formulas,
    summarize_batched,
    summarizer_info,
    is_formula_query,
)
from api.services.embed import embedding_info
from api.services import agent
from api.services import memory as mem
from api.services import websearch
from api.core.config import load_config


ProgressFn = Optional[Callable[[str], None]]
PctFn = Optional[Callable[[int], None]]
CancelFn = Optional[Callable[[], bool]]


class PipelineCancelled(RuntimeError):
    """Raised when a pipeline operation is cancelled."""


def _emit(callback: ProgressFn, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass


def _emit_pct(callback: PctFn, value: int) -> None:
    if not callback:
        return
    try:
        callback(value)
    except Exception:
        pass


def _check_cancel(should_cancel: CancelFn) -> None:
    if should_cancel and should_cancel():
        raise PipelineCancelled("Cancelled")


def ingest_documents(
    paths: Sequence[pathlib.Path],
    progress: ProgressFn = None,
    progress_pct: PctFn = None,
    should_cancel: CancelFn = None,
    *,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest documents into the vector index and produce summaries."""

    if not paths:
        return {"details": [], "num_docs": 0, "num_chunks": 0, "doc_summary": ""}

    summaries: List[str] = []
    details: List[Dict[str, Any]] = []
    total_chunks = 0

    info = embedding_info()
    _emit(progress, f"Embedding: {info['backend']} ({info['model']})")
    sumi = summarizer_info()
    _emit(progress, f"Summarizer: {sumi['backend']}")

    for raw in paths:
        _check_cancel(should_cancel)
        path = pathlib.Path(raw)
        _emit(progress, f"Loading {path.name}…")
        _emit_pct(progress_pct, 2)
        data = path.read_bytes()

        _emit(progress, "Parsing document…")
        parsed = parse_any_bytes(path.name, data, progress_cb=lambda msg: _emit(progress, msg))
        _check_cancel(should_cancel)
        _emit_pct(progress_pct, 15)

        ocr_list = parsed.get("ocr_page_numbers") or []
        suspect_list = parsed.get("suspect_pages") or []
        salv_list = parsed.get("salvaged_pages") or []

        if ocr_list:
            head = ", ".join(str(x) for x in ocr_list[:20])
            tail = " ..." if len(ocr_list) > 20 else ""
            _emit(progress, f"OCR pages: [{head}]{tail}")
        else:
            _emit(progress, "OCR pages: []")
        if suspect_list:
            head = ", ".join(str(x) for x in suspect_list[:20])
            tail = " ..." if len(suspect_list) > 20 else ""
            _emit(progress, f"Suspect pages: [{head}]{tail}")
        else:
            _emit(progress, "Suspect pages: []")
        if salv_list:
            head = ", ".join(str(x) for x in salv_list[:20])
            tail = " ..." if len(salv_list) > 20 else ""
            _emit(progress, f"Salvaged pages: [{head}]{tail}")
        else:
            _emit(progress, "Salvaged pages: []")

        for page in parsed["pages"]:
            page["doc_name"] = path.name

        _check_cancel(should_cancel)
        _emit(progress, "Chunking…")
        chunks = chunk_pages(parsed["pages"])
        _check_cancel(should_cancel)
        _emit_pct(progress_pct, 35)

        def embed_cb(done: int, total: int) -> None:
            _check_cancel(should_cancel)
            base = 35
            span = 55
            pct = base + int(span * (done / max(1, total)))
            pct = min(95, max(36, pct))
            _emit_pct(progress_pct, pct)

        _emit(progress, "Embedding and indexing…")
        add_chunks(chunks, progress_cb=embed_cb, user_id=user_id)
        _check_cancel(should_cancel)
        _emit_pct(progress_pct, 96)

        total_chunks += len(chunks)

        _emit(progress, "Summarizing…")
        _check_cancel(should_cancel)
        docsum = summarize_document(chunks)
        _emit_pct(progress_pct, 100)

        summaries.append(f"## {path.name}\n\n{docsum['summary']}")
        details.append(
            {
                "file": str(path),
                "num_pages": parsed["num_pages"],
                "ocr_pages": parsed["ocr_pages"],
                "ocr_page_numbers": ocr_list,
                "suspect_pages": suspect_list,
                "num_chunks": len(chunks),
            }
        )

    combined = "\n\n".join(summaries) if summaries else "No documents ingested."
    return {
        "details": details,
        "num_docs": len(paths),
        "num_chunks": total_chunks,
        "doc_summary": combined,
    }


def _token_overlap(q_text: str, chunk_text: str) -> float:
    qtoks = [t for t in re.findall(r"[A-Za-z0-9_]+", (q_text or "").lower()) if len(t) > 2]
    if not qtoks:
        return 0.0
    ctoks = [t for t in re.findall(r"[A-Za-z0-9_]+", (chunk_text or "").lower()) if len(t) > 2]
    if not ctoks:
        return 0.0
    cset = Counter(ctoks)
    shared = sum(1 for t in qtoks if cset.get(t, 0) > 0)
    return shared / max(1, len(qtoks))


def build_actions(ans_text: str, meta: Dict[str, Any], web_enabled: bool, has_openai: bool, only_doc: Optional[str], strict_docs: bool, attach_chunks: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Construct the actions array for UI from answer text and context flags.

    Kept as a module-level function so other modules (routes/UI) can import it.
    """
    actions: List[Dict[str, Any]] = []
    low = (ans_text or "").lower()
    has_next = bool(re.search(r"next steps", low))
    word_count = len(re.findall(r"\w+", ans_text or ""))
    suggest_broaden = bool(meta.get('suggest_broaden')) or bool(only_doc) or bool(strict_docs)

    actions.append({
        "id": "enable_web",
        "label": "Regenerate with Web Search",
        "description": "Re-run with live web search enabled alongside local documents.",
        "available": (not web_enabled) and has_openai,
        "recommended": (not web_enabled) and (not has_next or meta.get('not_found', False)),
        "requires_confirmation": True,
    })

    actions.append({
        "id": "broaden_docs",
        "label": "Broaden to all documents",
        "description": "Expand the search to every indexed document instead of a single selected document.",
        "available": bool(only_doc) or bool(strict_docs),
        "recommended": suggest_broaden,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "upload_docs",
        "label": "Upload / Attach Documents",
        "description": "Attach documents or pages relevant to this question so the assistant can cite them directly.",
        "available": True,
        "recommended": meta.get('not_found', False) and not (attach_chunks and len(attach_chunks) > 0),
        "requires_confirmation": True,
    })

    actions.append({
        "id": "expand_detail",
        "label": "Expand Answer Details",
        "description": "Produce a longer, more detailed answer and include more evidence and explicit next steps.",
        "available": True,
        "recommended": (word_count < 200) or (not has_next),
        "requires_confirmation": True,
    })

    actions.append({
        "id": "summarize_docs",
        "label": "Summarize Documents",
        "description": "Generate concise summaries of all relevant documents or the current document set.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "generate_quiz",
        "label": "Generate Quiz",
        "description": "Create quiz questions based on the answer content to test understanding.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "compare_sources",
        "label": "Compare Sources",
        "description": "Analyze and compare different perspectives or data from multiple sources.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "simplify_explanation",
        "label": "Simplify Explanation",
        "description": "Rephrase the answer in simpler terms, avoiding jargon and technical language.",
        "available": has_openai,
        "recommended": (word_count > 300) and has_openai,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "cite_sources_only",
        "label": "List Citations",
        "description": "Extract and list only the sources and citations from the answer.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "mindmap_outline",
        "label": "Create Mind Map",
        "description": "Generate a hierarchical outline or mind map structure of the answer.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "followup_questions",
        "label": "Suggest Follow-ups",
        "description": "Generate relevant follow-up questions to deepen understanding.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "translate_answer",
        "label": "Translate Answer",
        "description": "Translate the answer to another language while preserving technical accuracy.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "debug_reasoning_trace",
        "label": "Show Reasoning",
        "description": "Display detailed reasoning steps and decision-making process.",
        "available": has_openai,
        "recommended": False,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "detect_gaps",
        "label": "Identify Knowledge Gaps",
        "description": "Analyze what information is missing or incomplete in the current answer.",
        "available": has_openai,
        "recommended": (meta.get('not_found', False) or (word_count < 150)) and has_openai,
        "requires_confirmation": True,
    })

    actions.append({
        "id": "recommend_new_docs",
        "label": "Recommend Documents",
        "description": "Suggest additional documents or resources to upload for better answers.",
        "available": has_openai,
        "recommended": meta.get('not_found', False) and has_openai,
        "requires_confirmation": True,
    })

    return actions


def extract_next_steps(ans_text: str) -> List[str]:
    """Extract short next-step strings from an answer text.

    This is a public helper other modules can call.
    """
    if not ans_text:
        return []
    
    # Simple approach: look for lines that start with ** and contain a colon
    lines = ans_text.split('\n')
    steps = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('**') and ':' in line:
            # Extract the text between ** and :, and everything after :
            parts = line.split(':', 1)
            if len(parts) == 2:
                title = parts[0].replace('**', '').strip()
                desc = parts[1].strip()
                if title and desc:
                    steps.append(f"{title}: {desc}")
    
    # If no ** patterns found, try the original approach
    if not steps:
        txt = ans_text
        low = txt.lower()
        
        # Look for various patterns that indicate next steps
        patterns = [
            r"next\s*steps",
            r"suggested\s*next\s*steps",
            r"recommended\s*next\s*steps",
            r"you\s+can",
            r"you\s+should",
            r"try\s+",
            r"consider\s+",
            r"suggest\s+",
            r"recommend\s+"
        ]
        
        idx = -1
        for pattern in patterns:
            match = re.search(pattern, low)
            if match:
                idx = match.start()
                break
        
        if idx != -1:
            tail = txt[idx:]
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            
            # Skip the header line if it matches a pattern
            if lines and re.search(r"(next\s*steps|suggested|recommended|you\s+can|you\s+should|try|consider|suggest|recommend)", lines[0], re.I):
                lines = lines[1:]
            
            for ln in lines:
                # Match bullet points
                m = re.match(r"^[-•*\u2022]\s*(.+)$", ln)
                if m:
                    steps.append(m.group(1).strip())
                    continue
                # Match numbered lists
                m = re.match(r"^\d+[\).]\s*(.+)$", ln)
                if m:
                    steps.append(m.group(1).strip())
                    continue
                # Match text that looks like a step
                s = ln.strip()
                if len(s) < 200 and len(s.split()) <= 50:
                    # Stop if we hit a header or section break
                    if re.match(r"^#{1,3}\s*", s) or re.match(r"^[A-Z][A-Za-z ]+:$", s):
                        break
                    # Only add if it looks like a step (contains action words or is reasonably short)
                    if (re.search(r"(check|review|clarify|upload|enable|expand|try|consider|suggest)", s.lower()) or 
                        len(s) < 100):
                        steps.append(s)
                if len(steps) >= 8:
                    break
    
    cleaned = [re.sub(r"\s+", " ", s).strip(" \t\n\r\u2028\u2029") for s in steps]
    return cleaned


def generate_next_steps_with_llm(question: str, answer_text: str, trace: Dict[str, Any], meta: Dict[str, Any], *, web_enabled: bool, has_openai: bool, only_doc: Optional[str], strict_docs: bool, attach_chunks: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
    """Use an LLM to generate contextual next step suggestions based on the question, answer, and available tools.
    
    Returns a list of actionable next step suggestions that relate to available tools/actions.
    """
    import json
    try:
        # Only attempt when an OpenAI key/config is present
        cfg = load_config() or {}
        key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI key configured")

        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"

        # Available tools/actions context
        available_tools = []
        if not web_enabled and has_openai:
            available_tools.append("Enable web search to get current information")
        if only_doc or strict_docs:
            available_tools.append("Broaden search to include all documents")
        available_tools.extend([
            "Upload more documents for better context",
            "Expand the answer with more detailed information",
            "Generate a formula sheet if this is a technical topic",
            "Focus search on a specific document",
            "Adjust the search depth (top-k parameter)",
            "Toggle document reranking for better relevance",
            "Switch to RAG-only mode for document-focused search",
            "Perform an exhaustive search across all sources"
        ])

        # Build context for the LLM
        context = {
            "question": question,
            "answer_snippet": (answer_text or "")[:2000],  # Truncate for token limits
            "available_tools": available_tools,
            "current_settings": {
                "web_enabled": web_enabled,
                "only_doc": only_doc,
                "strict_docs": strict_docs,
                "has_attachments": bool(attach_chunks)
            }
        }

        # Available actions that can be executed
        available_actions = [
            {"id": "enable_web", "label": "Regenerate with Web Search", "description": "Re-run with live web search enabled"},
            {"id": "upload_docs", "label": "Upload Documents", "description": "Upload additional documents for better context"},
            {"id": "expand_detail", "label": "Expand Answer", "description": "Get a more detailed answer with more evidence"},
            {"id": "generate_formula_sheet", "label": "Generate Formula Sheet", "description": "Create a concise formula sheet"},
            {"id": "broaden_docs", "label": "Broaden Search", "description": "Search across all documents"},
            {"id": "select_doc", "label": "Focus on Document", "description": "Search within a specific document"},
            {"id": "set_top_k", "label": "Adjust Search Depth", "description": "Change the number of documents to search"},
            {"id": "toggle_reranker", "label": "Regenerate with Reranker", "description": "Re-run with document reranking enabled"},
            {"id": "strict_docs_only", "label": "Regenerate RAG-Only", "description": "Search only within uploaded documents"},
            {"id": "exhaustive_search", "label": "Regenerate Exhaustive", "description": "Perform a more thorough search"},
            {"id": "summarize_docs", "label": "Summarize Documents", "description": "Generate concise summaries of relevant documents"},
            {"id": "generate_quiz", "label": "Generate Quiz", "description": "Create quiz questions to test understanding"},
            {"id": "compare_sources", "label": "Compare Sources", "description": "Analyze different perspectives from multiple sources"},
            {"id": "simplify_explanation", "label": "Simplify Explanation", "description": "Rephrase in simpler terms without jargon"},
            {"id": "cite_sources_only", "label": "List Citations", "description": "Extract and list only the sources and citations"},
            {"id": "mindmap_outline", "label": "Create Mind Map", "description": "Generate hierarchical outline or mind map structure"},
            {"id": "followup_questions", "label": "Suggest Follow-ups", "description": "Generate relevant follow-up questions"},
            {"id": "translate_answer", "label": "Translate Answer", "description": "Translate answer to another language"},
            {"id": "debug_reasoning_trace", "label": "Show Reasoning", "description": "Display detailed reasoning steps and process"},
            {"id": "detect_gaps", "label": "Identify Knowledge Gaps", "description": "Analyze missing or incomplete information"},
            {"id": "recommend_new_docs", "label": "Recommend Documents", "description": "Suggest additional documents to upload"}
        ]

        action_list = [action['id'] for action in available_actions]
        system_prompt = (
            "You are an AI assistant that suggests relevant next steps for users based on their question and the current answer. "
            "You MUST select 3-5 action IDs from the list below and return them as a JSON array. "
            "Return ONLY a valid JSON array of action IDs, with no other text, explanation, or markdown formatting. "
            "Example: [\"enable_web\", \"upload_docs\", \"expand_detail\"]\n\n"
            f"AVAILABLE ACTION IDS (choose 3-5 from this list):\n{json.dumps(action_list, indent=2)}"
        )
        
        print(f"[DEBUG] Next steps system prompt: {system_prompt}")

        user_prompt = (
            f"Based on this question and answer, suggest relevant next steps:\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer_text[:1000]}...\n\n"
            f"Available tools: {', '.join(available_tools[:5])}\n\n"
            f"Generate 3-5 specific next step suggestions as a JSON array."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        raw_response = (response.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM raw response for next_steps: {raw_response}")
        
        # Parse JSON response
        try:
            # Clean the response first - remove markdown code blocks
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            next_steps = json.loads(cleaned_response)
            if isinstance(next_steps, list):
                # Validate that these are valid action IDs
                valid_action_ids = {action['id'] for action in available_actions}
                cleaned_steps = []
                for step in next_steps:
                    if isinstance(step, str) and step.strip():
                        cleaned = step.strip().strip('"').strip("'")  # Remove quotes
                        # Drop any clarifying variants that might slip through (e.g., "clarify_answer")
                        if cleaned.lower().startswith('clarify_'):
                            print(f"[DEBUG] Filtering out clarifying variant: {cleaned}")
                            continue
                        if cleaned in valid_action_ids:
                            cleaned_steps.append(cleaned)
                        else:
                            print(f"[DEBUG] Skipping invalid action ID: {cleaned}")
                print(f"[DEBUG] Parsed next_steps: {cleaned_steps}")
                return cleaned_steps[:5]  # Limit to 5 suggestions
        except json.JSONDecodeError:
            pass

        # Fallback: extract from the raw response if JSON parsing fails
        lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
        fallback_steps = []
        for line in lines:
            # Skip markdown code blocks
            if line.startswith('```'):
                continue
            # Remove common prefixes and clean up
            cleaned = re.sub(r'^[-*•\d+\.\s]*', '', line).strip()
            # Remove quotes and commas from malformed JSON
            cleaned = cleaned.strip('"').strip("'").strip(',').strip()
            if cleaned and len(cleaned) > 5 and len(cleaned) < 100:
                # Filter out any clarifying themed suggestions
                if 'clarify' in cleaned.lower():
                    continue
                fallback_steps.append(cleaned)
        return fallback_steps[:5]

    except Exception as e:
        # Fallback to regex-based extraction if LLM fails
        steps = extract_next_steps(answer_text)
        # If still no steps, provide some default action IDs
        if not steps:
            steps = [
                "upload_docs",
                "enable_web",
                "expand_detail"
            ]
        return steps


def generate_actions_with_llm(question: str, answer_text: str, trace: Dict[str, Any], meta: Dict[str, Any], *, web_enabled: bool, has_openai: bool, only_doc: Optional[str], strict_docs: bool, attach_chunks: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Use an LLM to generate a JSON array of actions given question, answer, and context.

    Falls back to build_actions(...) on any error or when no OpenAI key is present.
    """
    try:
        # Only attempt when an OpenAI key/config is present
        cfg = load_config() or {}
        key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI key configured")

        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"

        # Enhanced canonical actions we support — the LLM may select from these or suggest params
        canonical = [
            {"id": "enable_web", "label": "Enable Web Search", "description": "Run live Web search alongside local documents."},
            {"id": "broaden_docs", "label": "Broaden to all documents", "description": "Search across all indexed documents."},
            {"id": "upload_docs", "label": "Upload / Attach Documents", "description": "Attach documents/pages for more context."},
            {"id": "expand_detail", "label": "Expand Answer Details", "description": "Produce a longer, more detailed answer and include more evidence."},
            {"id": "generate_formula_sheet", "label": "Generate Formula Sheet", "description": "Create a concise list of formulas relevant to the question."},
            {"id": "select_doc", "label": "Focus on Specific Document", "description": "Search within a specific document for more targeted results."},
            {"id": "set_top_k", "label": "Adjust Search Depth", "description": "Change the number of documents to search through."},
            {"id": "toggle_reranker", "label": "Toggle Reranker", "description": "Enable/disable document reranking for better relevance."},
            {"id": "strict_docs_only", "label": "RAG Only Mode", "description": "Search only within uploaded documents, no web search."},
            {"id": "exhaustive_search", "label": "Exhaustive Search", "description": "Perform a more thorough search across all sources."},
        ]

        avail_map = {
            "enable_web": (not web_enabled) and has_openai,
            "broaden_docs": bool(only_doc) or bool(strict_docs),
            "upload_docs": True,
            "expand_detail": True,
            "generate_formula_sheet": True,
            "select_doc": True,
            "set_top_k": True,
            "toggle_reranker": True,
            "strict_docs_only": not strict_docs,
            "exhaustive_search": True,
        }

        safe_trace = {k: trace.get(k) for k in ("retrieval", "selected_context", "web_sources") if trace and trace.get(k) is not None}
        payload = {
            "question": question,
            "answer_snippet": (answer_text or "")[:4000],
            "trace": safe_trace,
            "meta": meta or {},
            "canonical_actions": canonical,
            "availability": avail_map,
        }

        system = (
            "You are an assistant that outputs a strict JSON array of suggested UI actions. "
            "Each action must be a JSON object with keys: id, label, description, available (bool), recommended (bool), params (object or null), requires_confirmation (bool). "
            "Only use ids from the provided canonical_actions unless an explicit param mapping is provided. "
            "If there are no reasonable actions, return an empty array: [] and nothing else."
        )
        user_msg = (
            "Given the user question, the assistant's answer, and retrieval trace/context, return a JSON array of suggested actions. "
            "Only output valid JSON. Do not include any explanatory text.\n\n"
            f"Context payload:\n{json.dumps(payload, default=str)[:2500]}"
        )

        msg = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = (msg.choices[0].message.content or "").strip()
        import json as _json
        parsed = _json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("LLM did not return a list")

        # Validate and normalize actions
        validated: List[Dict[str, Any]] = []
        allowed_ids = {c["id"] for c in canonical}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("id") or "").strip()
            if not aid:
                continue
            if aid not in allowed_ids:
                # ignore unknown ids for safety
                continue
            label = str(item.get("label") or next((c["label"] for c in canonical if c["id"] == aid), aid))
            desc = str(item.get("description") or next((c["description"] for c in canonical if c["id"] == aid), ""))
            available = bool(item.get("available")) if ("available" in item) else bool(avail_map.get(aid, False))
            recommended = bool(item.get("recommended")) if ("recommended" in item) else False
            params = item.get("params") if isinstance(item.get("params"), dict) else None
            requires_confirmation = bool(item.get("requires_confirmation")) if ("requires_confirmation" in item) else True
            validated.append({
                "id": aid,
                "label": label,
                "description": desc,
                "available": available,
                "recommended": recommended,
                "params": params,
                "requires_confirmation": requires_confirmation,
            })

        # Ensure at least one action is present; otherwise fallback to heuristics
        if not validated:
            return build_actions(answer_text, meta, web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=strict_docs, attach_chunks=attach_chunks)
        return validated
    except Exception:
        # Fallback to heuristic builder
        try:
            return build_actions(answer_text, meta, web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=strict_docs, attach_chunks=attach_chunks)
        except Exception:
            return []


def answer_question(
    question: str,
    *,
    k: int = 5,
    history: Optional[Sequence[Dict[str, str]]] = None,
    formula_mode: bool = False,
    agents_enabled: bool = False,
    web_enabled: bool = False,
    strict_docs: bool = False,
    progress: ProgressFn = None,
    progress_pct: PctFn = None,
    should_cancel: CancelFn = None,
    only_doc: Optional[str] = None,
    user_id: Optional[str] = None,
    exhaustive: Optional[bool] = None,
    extra_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the question-answering pipeline synchronously."""

    _check_cancel(should_cancel)

    # Keep the original user phrasing for final rendering
    _orig_user_q = question

    # Build formatting prefix conservatively — never mention LaTeX or Markdown in the answer.
    plain_prefix = (
        "Answer clearly and professionally in plain text. "
        "Do not discuss formatting, Markdown, or LaTeX. "
        "Preserve citations as plain text when applicable.\n"
    )
    fmt_prefix = plain_prefix + "\n"
    # If formula mode or math-y question, add concise one-line formula list instruction without LaTeX mention
    if formula_mode or is_formula_query(question):
        fmt_prefix += (
            "When listing formulas, format EACH item on ONE line exactly as: "
            "<label/meaning> — <formula> [FileName.pdf p.N] — <1–2 sentence explanation>. "
            "Put the citation AFTER the formula. Keep explanations concise and factual.\n\n"
        )
    # Strong guidance: prefer detailed, evidence-based answers
    # NOTE: We generate next steps separately using a dedicated LLM call, so we don't want them in the answer
    fmt_prefix += (
        "Prioritize a thorough, evidence-backed answer. Begin with a one-line summary, then provide detailed supporting sections or bullets. "
        "Aim for ~250–600 words unless the user asked for brevity. DO NOT include a 'Next steps' section - that will be generated separately.\n\n"
    )
    fmt_q = fmt_prefix + question

    history = list(history or [])

    # Local helper to ensure any early returns include actions and next_steps
    def _wrap_out(out_obj: Dict[str, Any]) -> Dict[str, Any]:
        try:
            meta_local = out_obj.get('meta') if isinstance(out_obj.get('meta'), dict) else {}
        except Exception:
            meta_local = {}
        answer_text_local = out_obj.get('answer') or ''
        try:
            heur = build_actions(answer_text_local, meta_local, web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=bool(strict_docs), attach_chunks=attach_chunks)
        except Exception:
            heur = []
        final = heur
        try:
            if has_openai:
                try:
                    llm_a = generate_actions_with_llm(_orig_user_q or question, answer_text_local, {}, meta_local, web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=bool(strict_docs), attach_chunks=attach_chunks)
                    if isinstance(llm_a, list) and llm_a:
                        # merge
                        m = {a.get('id'): dict(a) for a in heur}
                        for a in llm_a:
                            aid = a.get('id')
                            if not aid:
                                continue
                            if aid in m:
                                m[aid].update({k: v for k, v in a.items() if v is not None})
                            else:
                                m[aid] = dict(a)
                        final = list(m.values())
                except Exception:
                    pass
        except Exception:
            pass
        out_obj['actions'] = final or []
        try:
            out_obj['next_steps'] = extract_next_steps(answer_text_local)
        except Exception:
            out_obj['next_steps'] = []
        return out_obj
    q_lower = (question or "").strip().lower()
    os.environ["ASK_FORMULA_MODE"] = "true" if formula_mode else "false"
    if history:
        last = history[-1]
        if re.search(r"what\s+(?:was|is)\s+(?:my|the)\s+(?:last|previous)\s+question", q_lower):
            prev_q = last.get("q") or "(unknown)"
            text = f"Your previous question was:\n\n> {prev_q}"
            return _wrap_out({"answer": text, "citations": [], "quotes": []})
        if re.search(r"what\s+(?:was|is)\s+(?:your|the)\s+(?:last|previous)\s+answer", q_lower) or re.search(
            r"repeat\s+your\s+(?:last|previous)\s+answer", q_lower
        ):
            prev_a = last.get("a") or "(no recent answer recorded)"
            text = "Here is my previous answer:\n\n" + prev_a
            return _wrap_out({"answer": text, "citations": [], "quotes": []})
    try:
        q_pronoun_rx = re.compile(r"\b(it|they|them|that|this|these|those|he|she|him|her|its)\b", re.I)
        q_has_pronoun = bool(q_pronoun_rx.search(_orig_user_q))

        def _extract_referent_from_history(hist):
            if not hist:
                return None
            cand_scores = {}
            fname_rx = re.compile(r"([\w\- ]+\.(?:pdf|docx|pptx|txt))", re.I)
            cap_rx = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
            for turn in hist[-6:]:
                for fld in (turn.get("q"), turn.get("a")):
                    if not fld:
                        continue
                    # filenames
                    for m in fname_rx.findall(fld):
                        k = m.strip()
                        cand_scores[k] = cand_scores.get(k, 0) + 3
                    # capitalized name phrases
                    for m in cap_rx.findall(fld):
                        # filter out single common words like 'The'
                        if len(m.split()) >= 1:
                            k = m.strip()
                            cand_scores[k] = cand_scores.get(k, 0) + 1
                    # long tokens (likely topics)
                    toks = re.findall(r"[A-Za-z0-9_]{6,}", fld)
                    for t in toks:
                        cand_scores[t] = cand_scores.get(t, 0) + 0.5
            if not cand_scores:
                return None
            # Prefer filenames and multi-word capitalized phrases, then reasonable single-word names
            sorted_cands = sorted(cand_scores.items(), key=lambda kv: -kv[1])
            stop_words = {"the","this","that","it","what","how","why","when","where","who"}
            # Build an anchor string from the original user phrasing and the last user question
            try:
                anchor = "".join([_orig_user_q or "", " ", (hist[-1].get("q") or "") if hist else ""]).lower()
            except Exception:
                anchor = (_orig_user_q or "").lower()
            anchor_matches = []
            verb_stop = {"included","learned","learn","attended","did","do","was","is","are","have","has","included"}
            for cand, _ in sorted_cands:
                c = cand.strip()
                if not c:
                    continue
                low = c.lower()
                # Whole-word match in anchor (prefer these but filter verbs)
                try:
                    if re.search(r"\b" + re.escape(low) + r"\b", anchor) and low not in stop_words:
                        anchor_matches.append(c)
                        continue
                except Exception:
                    pass
                if low in stop_words:
                    continue
                # Accept filenames immediately
                if "." in c and any(c.lower().endswith(ext) for ext in (".pdf",".docx",".pptx",".txt")):
                    return c
                # Accept multi-word capitalized phrases (likely proper names or titles)
                if " " in c and len(c.split()) >= 2:
                    return c
                # Accept single capitalized words that look like names (length >=4)
                if c[0].isupper() and len(c) >= 4 and c.isalpha():
                    return c
                # Accept longer tokens (topic-like)
                if len(c) >= 6 and re.match(r"^[A-Za-z0-9_\-]+$", c):
                    return c
            # Choose best anchor match if available (prefer longer tokens)
            if anchor_matches:
                # filter out obvious verbs
                filtered = [a for a in anchor_matches if a.lower() not in verb_stop]
                candidates = filtered if filtered else anchor_matches
                candidates.sort(key=lambda s: -len(s))
                return candidates[0]
            return None

        if history and q_has_pronoun:
            # Build a short context from the last up to 3 turns (most recent last)
            parts = []
            for turn in history[-3:]:
                pq = (turn.get("q") or "").strip()
                pa = (turn.get("a") or "").strip()
                if pq:
                    parts.append(f"Prev question: {pq}")
                if pa:
                    parts.append(f"Prev answer: {pa}")
            # Try to surface a referent label (filename, topic, or name) to help retrieval
            ref = _extract_referent_from_history(history)
            if ref:
                parts.append(f"Referent: {ref}")
            if parts:
                prefix = "\n".join(parts)
                # Use the original user phrasing as the follow-up to preserve intent
                question = f"{prefix}\nFollow-up: {_orig_user_q}"
        else:
            # If no pronoun detected, keep previous lightweight behavior: if last
            # turn had a question but no answer (e.g., cancelled), include it.
            if history:
                _last = history[-1]
                prev_q = (_last.get("q") or "").strip()
                prev_a = (_last.get("a") or "").strip()
                if prev_q and not prev_a:
                    question = f"{prev_q}\nFollow-up: {_orig_user_q}"
    except Exception:
        pass

    tb_base = int(os.environ.get("ASK_TIME_BUDGET_SEC", "120"))
    tb = int(os.environ.get("ASK_TIME_BUDGET_SEC_FORMULA", "240")) if formula_mode else tb_base
    web_chunks: List[Dict[str, Any]] = []
    web_hits: List[Any] = []
    max_web_overlap = 0.0
    provider_label = "duckduckgo"
    try:
        from api.services.websearch import get_provider_name
        provider_label = get_provider_name() or "duckduckgo"
    except Exception:
        pass

    if web_enabled:
        # Web search and strict docs are mutually exclusive; allow both sources by disabling strict
        if strict_docs:
            strict_docs = False
            _emit(progress, "Strict documents disabled due to Web search.")
        _emit(progress, f"Web search ({provider_label})…")
        try:
            web_results = websearch.search_web(question, max_results=6)
        except Exception:
            web_results = []
        if web_results:
            _emit(progress, f"Web results: {len(web_results)}")
        else:
            _emit(progress, "Web search returned no results.")
        from urllib.parse import urlparse

        for res in web_results:
            snippet = (res.get("snippet") or "").strip()
            title = (res.get("title") or "").strip()
            if not snippet and not title:
                continue
            text = (title + "\n" + snippet).strip()
            url = res.get("url") or ""
            host = urlparse(url).netloc or (res.get("source") or "web")
            chunk = {
                "text": text,
                "doc_name": f"Web:{host}",
                "page_start": 1,
                "page_end": 1,
                "section_tag": "web",
                "web_url": url,
                "is_web": True,
                "_score": 0.85,
            }
            max_web_overlap = max(max_web_overlap, _token_overlap(question, text))
            web_chunks.append(chunk)
            web_hits.append((0.85, chunk))

    doc_reference = bool(re.search(r"(textbook|chapter|section|lecture|notes|\.(pdf|docx))", q_lower))

    rel_score, rel_meta = agent.estimate_relevance(question)
    _emit(progress, f"Relevance score: {rel_score if rel_score is not None else 'n/a'}")
    # If OpenAI is not configured, degrade advanced features to reduce costs and encourage OpenAI usage
    try:
        from api.core.config import load_config as _cfg
        __cfg = _cfg() or {}
        has_openai = bool(__cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    except Exception:
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if not has_openai:
        if web_enabled:
            _emit(progress, "Web search disabled (OpenAI key not set).")
        web_enabled = False
        agents_enabled = False
        # Shorten time budgets for local generation
        try:
            tb = min(tb, int(os.environ.get("ASK_TIME_BUDGET_SEC_HF", "80")))
        except Exception:
            pass
    # If RAG-only (strict_docs) is requested and relevance is low, avoid LaTeX guidance entirely
    try:
        strict_min = float(os.getenv("ASK_STRICT_MIN_RELEVANCE", "0.15"))
    except Exception:
        strict_min = 0.15
    if strict_docs and (rel_score is not None) and (rel_score < strict_min):
        fmt_prefix = plain_prefix + "\n"
        fmt_q = fmt_prefix + question
    # Decide if we should also include local RAG when web is enabled.
    # Use the same LLM judge logic as strict mode, but as a gate (lenient in that any >0/Relevant passes).
    lenient_th = float(os.getenv("ASK_WEB_LENIENT_MIN", "0.05"))
    include_rag = True if not web_enabled else False
    hits: List[Any] = []
    if web_enabled:
        include_rag = bool(doc_reference or only_doc)
        # Heuristic hint before LLM
        if not include_rag and rel_score is not None:
            include_rag = (rel_score >= lenient_th)
        # Run a light local search to feed the LLM gate
        try:
            gate_pool = int(os.getenv("ASK_WEB_GATE_CANDIDATES", "80"))
            st_gate = int(os.getenv("ASK_WEB_GATE_TIMEOUT", "12"))
            _emit(progress, "Local gate search…")
            hits = search(
                question,
                k=gate_pool,
                progress_cb=lambda s: _emit(progress, s),
                timeout_sec=st_gate,
                pool=gate_pool,
                only_doc=only_doc,
                user_id=user_id,
            )
            gate_max = int(os.getenv("ASK_STRICT_LLM_MAX_HITS", "24"))
            gate_chunks = []
            for sc, ch in hits[:max(1, min(gate_max, len(hits)))] if hits else []:
                x = dict(ch); x["_score"] = float(sc); gate_chunks.append(x)
            if gate_chunks:
                llm_score, meta = agent.llm_relevance_gate(question, gate_chunks, doc_focus=only_doc)
                decided = None
                llm_label = meta.get('label') if isinstance(meta, dict) else None
                if isinstance(llm_label, str):
                    decided = (llm_label.strip().lower() == 'relevant')
                elif llm_score is not None:
                    decided = (float(llm_score) > 0.0)
                if progress:
                    try:
                        detail = ""
                        if isinstance(meta, dict) and meta.get('ctx_snippets') is not None:
                            detail = f" • ctx={meta.get('ctx_snippets')} snippets/{meta.get('ctx_chars')} chars"
                        progress(f"LLM web gate: score={llm_score if llm_score is not None else 'n/a'} label={llm_label or 'n/a'}{detail}")
                    except Exception:
                        pass
                if decided is True:
                    include_rag = True
        except Exception:
            pass

    rag_chunks: List[Dict[str, Any]] = []
    attach_chunks: List[Dict[str, Any]] = list(extra_chunks or [])
    mem_chunks: List[Dict[str, Any]] = []
    rag_scores: List[float] = []

    if (not web_enabled) or include_rag:
        base_pool = int(os.environ.get("ASK_CANDIDATES", "300"))
        pool = int(os.environ.get("ASK_CANDIDATES_FORMULA", "3000")) if formula_mode else base_pool
        pc_holder = {"value": 12}

        def s_cb(msg: str) -> None:
            _emit(progress, msg)
            pc_holder["value"] = min(55, pc_holder["value"] + 3)
            _emit_pct(progress_pct, pc_holder["value"])

        st_base = max(10, min(tb_base // 2, 30))
        st = int(os.environ.get("SEARCH_TIMEOUT_SEC", str(st_base)))
        if formula_mode:
            st = int(os.environ.get("SEARCH_TIMEOUT_SEC_FORMULA", str(max(20, min(tb // 2, 60)))))

        _check_cancel(should_cancel)
        _emit(progress, "Searching index…")
        if not hits:
            hits = search(
                question,
                k=pool,
                progress_cb=s_cb,
                timeout_sec=st,
                pool=pool,
                only_doc=only_doc,
                user_id=user_id,
            )
        if only_doc and not hits:
            _emit(progress, "No hits in selected document; expanding to all documents…")
            hits = search(
                question,
                k=pool,
                progress_cb=s_cb,
                timeout_sec=st,
                pool=pool,
                user_id=user_id,
            )
        _check_cancel(should_cancel)
        _emit(progress, f"Hits: {len(hits)}")
        rag_scores = [float(h[0]) for h in hits]
        for sc, ch in hits[:k]:
            x = dict(ch)
            x["_score"] = float(sc)
            rag_chunks.append(x)
        if not web_enabled and not hits:
            # If there are attached chunks or memory chunks, continue using them as
            # context instead of failing immediately. Only return a no-results
            # message when there truly is no context to answer from.
            if not (attach_chunks or mem_chunks):
                _emit_pct(progress_pct, 100)
                return _wrap_out({"answer": "**No local results. Try web search.**", "citations": [], "quotes": []})
    else:
        _emit(progress, "Skipping local RAG (lenient gate). Using web results.")
        if web_enabled and not web_chunks:
            _emit(progress, "Web search returned no results; falling back to local RAG…")
            try:
                # run a light search to avoid empty context
                st_base = max(8, min(tb_base // 3, 20))
                st = int(os.environ.get("SEARCH_TIMEOUT_SEC", str(st_base)))
                hits = search(
                    question,
                    k=int(os.environ.get("ASK_CANDIDATES", "150")),
                    progress_cb=lambda s: _emit(progress, s),
                    timeout_sec=st,
                    pool=int(os.environ.get("ASK_CANDIDATES", "150")),
                    only_doc=only_doc,
                    user_id=user_id,
                )
                _emit(progress, f"Fallback Hits: {len(hits)}")
                for sc, ch in hits[:k]:
                    x = dict(ch)
                    x["_score"] = float(sc)
                    rag_chunks.append(x)
            except Exception:
                pass

    rag_threshold = float(os.getenv("ASK_WEB_MIN_RAG", "0.35"))
    strict_answer_min = float(os.getenv("ASK_STRICT_MIN_RAG", "0.35"))

    if strict_docs:
        # Strict mode: use only retrieved document chunks; no web or memory
        top_chunks = rag_chunks
        # Numeric guardrails (fallback when LLM is unavailable)
        overlap_min = float(os.getenv("ASK_STRICT_MIN_OVERLAP", "0.18"))
        max_overlap = 0.0
        for c in top_chunks[:max(1, min(10, len(top_chunks)))]:
            try:
                max_overlap = max(max_overlap, _token_overlap(question, c.get("text") or ""))
            except Exception:
                pass
        # Keyword coverage across chunks
        def _q_tokens(s: str):
            toks = re.findall(r"[a-z0-9]{3,}", (s or '').lower())
            stop = {"what","which","when","where","why","how","does","do","is","are","was","were","can","could","should","would","will","might","may","the","and","for","from","this","that","these","those","into","about","your","our","their","its","of","to","in","on","at","by","it","we","you","i","a","an","plan","plans","feature","features","user","users","application","app","goal","goals"}
            return [t for t in toks if t not in stop]
        qtok = set(_q_tokens(question))
        covered = set()
        support_chunks = 0
        for c in top_chunks[:max(1, min(8, len(top_chunks)))]:
            txt = (c.get('text') or '').lower()
            if not txt:
                continue
            has = False
            for t in qtok:
                if t in txt:
                    covered.add(t)
                    has = True
            if has:
                support_chunks += 1
        min_kw = int(os.getenv("ASK_STRICT_MIN_KW", "2"))
        min_support = int(os.getenv("ASK_STRICT_MIN_SUPPORT", "1"))
        coverage_ok = (len(covered) >= min_kw)
        support_ok = (support_chunks >= min_support)
        numeric_ok = bool(rag_scores) and (rag_scores[0] >= strict_answer_min) and (max_overlap >= overlap_min)

        # LLM relevance adjudicator — primary decision if available
        llm_score = None
        llm_label = None
        llm_reason = None
        try:
            # Provide the judge with a wider slice of retrieved context than top-k
            gate_max = int(os.getenv("ASK_STRICT_LLM_MAX_HITS", "24"))
            gate_chunks = []
            for sc, ch in hits[:max(1, min(gate_max, len(hits)))] if hits else []:
                x = dict(ch)
                x["_score"] = float(sc)
                gate_chunks.append(x)
            src_for_gate = gate_chunks if gate_chunks else top_chunks
            llm_score, meta = agent.llm_relevance_gate(question, src_for_gate, doc_focus=only_doc)
            llm_label = meta.get('label') if isinstance(meta, dict) else None
            llm_reason = meta.get('reason') if isinstance(meta, dict) else None
            if progress:
                try:
                    lbl = f" {llm_label}" if llm_label else ""
                    detail = ""
                    if isinstance(meta, dict) and meta.get('ctx_snippets') is not None:
                        detail = f" • ctx={meta.get('ctx_snippets')} snippets/{meta.get('ctx_chars')} chars"
                    progress(f"LLM judge: {meta.get('model','?')} score={llm_score if llm_score is not None else 'n/a'}{lbl}{detail}")
                    if llm_reason:
                        progress(f"LLM reason: {llm_reason[:180]}")
                except Exception:
                    pass
            # Decision policy: use label if present, else treat 0.0 as Irrelevant, >0.0 as Relevant
            decided: Optional[bool] = None
            if isinstance(llm_label, str):
                decided = (llm_label.strip().lower() == 'relevant')
            elif llm_score is not None:
                decided = (float(llm_score) > 0.0)
            if decided is None:
                # No LLM score available – fall back to numeric checks
                if not (numeric_ok and coverage_ok and support_ok):
                    _emit(progress, f"Strict guard: rag={rag_scores[0] if rag_scores else 'n/a'} ov={max_overlap:.2f} kw={len(covered)} sup={support_chunks} (LLM n/a)")
                    _emit_pct(progress_pct, 100)
                    return _wrap_out({"answer": "**I couldn't find relevant information in your documents.**", "citations": [], "quotes": []})
            elif decided is False:
                _emit(progress, f"Strict guard: LLM veto  score={llm_score if llm_score is not None else 'n/a'} label={llm_label or 'n/a'}")
                _emit_pct(progress_pct, 100)
                msg = (
                    "**No directly relevant passage found.**\n\n"
                    "- Broaden to all documents\n"
                    "- Enable Web search\n"
                    "- Clarify or upload relevant pages"
                )
                return _wrap_out({"answer": msg, "citations": [], "quotes": []})
            else:
                _emit(progress, f"Strict guard: LLM pass — score={llm_score if llm_score is not None else 'n/a'} label={llm_label or 'n/a'}")
        except Exception:
            # On LLM failure, use numeric gates only
            if not (numeric_ok and coverage_ok and support_ok):
                _emit(progress, f"Strict guard: rag={rag_scores[0] if rag_scores else 'n/a'} ov={max_overlap:.2f} kw={len(covered)} sup={support_chunks} (LLM error)")
                _emit_pct(progress_pct, 100)
                msg = (
                    "**No directly relevant passage found.**\n\n"
                    "- Broaden to all documents\n"
                    "- Enable Web search\n"
                    "- Clarify or upload relevant pages"
                )
                return _wrap_out({"answer": msg, "citations": [], "quotes": []})
    elif web_enabled and web_chunks:
        if include_rag and rag_chunks:
            top_chunks = list(web_chunks) + rag_chunks
        else:
            top_chunks = list(web_chunks)
    else:
        top_chunks = rag_chunks

    # Prepend any extra chunks (e.g., attached image descriptions)
    if attach_chunks:
        try:
            top_chunks = list(attach_chunks) + list(top_chunks or [])
        except Exception:
            top_chunks = list(attach_chunks)

    # In-session memory (recent turns from current chat) — keep a longer window
    # Always include memory as context for pronoun/coreference resolution.
    if history:
        for idx, turn in enumerate(history[-16:], 1):
            q_prev = (turn.get("q") or "").strip()
            a_prev = (turn.get("a") or "").strip()
            if not q_prev and not a_prev:
                continue
            parts = []
            if q_prev:
                parts.append(f"Prev question: {q_prev}")
            if a_prev:
                parts.append(f"Prev answer: {a_prev}")
            text_mem = "\n".join(parts)
            if not text_mem:
                continue
            mem_chunks.append(
                {
                    "text": text_mem,
                    "doc_name": "Conversation memory",
                    "page_start": idx,
                    "page_end": idx,
                    "section_tag": "memory",
                    "is_memory": True,
                    "_score": 0.90,
                }
            )

    # Cross-conversation memory from memory file (always include as context)
    try:
        limit = int(os.getenv("MEMORY_TOKEN_LIMIT", "1200"))
        recent = mem.load_recent(limit_tokens=limit, user_id=user_id) or []
        # Deduplicate against current in-session history
        hist_set = {
            ((t.get("q") or "").strip(), (t.get("a") or "").strip())
            for t in (history or [])
        }
        cross = []
        for t in recent:
            pair = ((t.get("q") or "").strip(), (t.get("a") or "").strip())
            if not pair[0] and not pair[1]:
                continue
            if pair in hist_set:
                continue
            cross.append(t)
        # Keep a modest number to avoid flooding context
        for j, turn in enumerate(cross[-8:], 1):
            q_prev = (turn.get("q") or "").strip()
            a_prev = (turn.get("a") or "").strip()
            parts = []
            if q_prev:
                parts.append(f"Prev question: {q_prev}")
            if a_prev:
                parts.append(f"Prev answer: {a_prev}")
            text_mem = "\n".join(parts)
            if not text_mem:
                continue
            mem_chunks.append(
                {
                    "text": text_mem,
                    "doc_name": "Long‑term memory",
                    "page_start": j,
                    "page_end": j,
                    "section_tag": "memory",
                    "is_memory": True,
                    "_score": 0.85,
                }
            )
    except Exception:
        # Memory is best-effort; ignore failures
        pass
    if mem_chunks:
        try:
            top_chunks = list(mem_chunks) + list(top_chunks or [])
        except Exception:
            top_chunks = list(mem_chunks)

    # Start with whatever we have so far; refine per mode below
    combined_hits = list(hits) if hits else []
    if attach_chunks:
        try:
            combined_hits = [(0.95, ch) for ch in attach_chunks] + combined_hits
        except Exception:
            pass

    if strict_docs:
        combined_hits = list(hits)
    elif web_enabled:
        # Always include web chunks; optionally include rag if we ran it
        base_hits = list(hits) if include_rag else []
        web_extras = web_hits if web_hits else [(0.85, c) for c in web_chunks]
        combined_hits = base_hits + web_extras
    if mem_chunks:
        combined_hits.extend([(0.80, ch) for ch in mem_chunks])

    if formula_mode:
        _check_cancel(should_cancel)
        _emit(progress, "Formula mode: extracting all formulas in scope…")
        total = len(top_chunks)
        if total < (len(hits) if include_rag else len(web_chunks)):
            _emit(progress, f"Warning: only {total} chunks returned; consider increasing index size or candidate pool.")

        def p_cb(msg: str) -> None:
            _check_cancel(should_cancel)
            _emit(progress, msg)
            m = re.search(r"(\d+)\s*/\s*(\d+)", msg)
            if m:
                done = int(m.group(1))
                pct = 20 + int(70 * (done / max(1, total)))
                pct = max(20, min(95, pct))
                _emit_pct(progress_pct, pct)

        out = summarize_all_formulas(fmt_q, top_chunks, progress_cb=p_cb)
        if agents_enabled:
            _check_cancel(should_cancel)
            try:
                _emit(progress, "Agents: verifying…")
                agent_budget = int(os.environ.get("AGENT_VERIFY_BUDGET", "30"))
                _prev = out if isinstance(out, dict) else {"answer": str(out)}
                agent_hits = [
                    h[1] if isinstance(h, tuple) and len(h) > 1 else h
                    for h in (combined_hits or [])
                ]
                _res = agent.verify_answer(question, out, agent_hits, time_budget_sec=agent_budget)
                changed = False
                try:
                    changed = (_res.get("answer", "") != _prev.get("answer", "")) if isinstance(_res, dict) else False
                except Exception:
                    changed = False
                if isinstance(_res, dict):
                    meta = _res.get("agent_meta", {})
                    meta.update({"enabled": True, "changed": bool(changed)})
                    _res["agent_meta"] = meta
                out = _res
                _emit(progress, "Agents: verdict — " + ("modified" if changed else "validated"))
            except Exception:
                _emit(progress, "Agent verification skipped (error).")
            _check_cancel(should_cancel)
    else:
        _check_cancel(should_cancel)
        _emit(progress, "Summarizing with context…")
        _emit_pct(progress_pct, 60)
        mb = int(os.environ.get("ASK_MAX_BATCHES", "6"))
        ex_env = os.environ.get("ASK_EXHAUSTIVE", "false").lower() in {"1", "true", "yes", "on"}
        ex_flag = bool(exhaustive) if exhaustive is not None else ex_env
        # Dedupe similar/duplicate contexts (keep first occurrence per doc/page and near-identical text)
        def _dedupe(chs):
            seen = set(); out = []
            for c in chs:
                try:
                    key = (str(c.get('doc_name') or ''), int(c.get('page') or c.get('page_start') or 0))
                except Exception:
                    key = (str(c.get('doc_name') or ''), str(c.get('page') or c.get('page_start') or '?'))
                txt = (c.get('text') or '').strip()
                sig = (key, txt[:180])
                if sig in seen: continue
                seen.add(sig); out.append(c)
            return out
        chs = _dedupe(list(top_chunks))
        _check_cancel(should_cancel)
        out = summarize_batched(
            fmt_q,
            chs,
            history=history,
            progress_cb=lambda s: _emit(progress, s),
            max_batches=mb,
            time_budget_sec=tb,
            exhaustive=ex_flag,
            strict_docs=strict_docs,
            orig_question=_orig_user_q,
            allow_not_found=(False if (web_enabled or (attach_chunks and len(attach_chunks) > 0)) else True),
        )
        if agents_enabled:
            _check_cancel(should_cancel)
            try:
                _emit(progress, "Agents: verifying…")
                agent_budget = int(os.environ.get("AGENT_VERIFY_BUDGET", "30"))
                _prev = out if isinstance(out, dict) else {"answer": str(out)}
                agent_hits = [
                    h[1] if isinstance(h, tuple) and len(h) > 1 else h
                    for h in (combined_hits or [])
                ]
                _res = agent.verify_answer(question, out, agent_hits, time_budget_sec=agent_budget)
                changed = False
                try:
                    changed = (_res.get("answer", "") != _prev.get("answer", "")) if isinstance(_res, dict) else False
                except Exception:
                    changed = False
                if isinstance(_res, dict):
                    meta = _res.get("agent_meta", {})
                    meta.update({"enabled": True, "changed": bool(changed)})
                    _res["agent_meta"] = meta
                out = _res
                _emit(progress, "Agents: verdict — " + ("modified" if changed else "validated"))
            except Exception:
                _emit(progress, "Agent verification skipped (error).")
            _check_cancel(should_cancel)

    try:
        if isinstance(out, dict) and "citations" in out:
            out["citations"] = sorted(set(out.get("citations", [])))
    except Exception:
        pass

    _emit_pct(progress_pct, 100)

    try:
        if isinstance(out, dict):
            ans = (out.get("answer") or "").strip()
            low = ans.lower()
            needs_md = ("**" not in ans) and ("- " not in ans)
            not_found = ("couldn't find relevant" in low) or ("couldn’t find relevant" in low) or ("no directly relevant" in low)
            if not_found and needs_md:
                bullets = (
                    "**No directly relevant passage found.**\n\n"
                    "- Broaden to all documents\n"
                    "- Enable Web search\n"
                    "- Clarify or upload relevant pages"
                )
                try:
                    ql = (question or "").lower()
                    if any(k in ql for k in ("strength", "weakness", "compare", "versus", "vs ", "trade-off", "tradeoff")):
                        bullets += "\n\n**General perspective**\n- Typical CS strengths: strong fundamentals, algorithms/data structures, code quality\n- Typical gaps vs seasoned domain profiles: domain context, production scale experience, stakeholder comms"
                except Exception:
                    pass
                out["answer"] = bullets
    except Exception:
        pass

    try:
        def _brief(c: Dict[str, Any]) -> Dict[str, Any]:
            try:
                doc = c.get("doc_name") or "?"
                ps = c.get("page_start") or c.get("page")
                pe = c.get("page_end") or ps
                page = int(ps) if isinstance(ps, int) else ps
                if pe and pe != ps:
                    page = f"{ps}-{pe}"
            except Exception:
                page = c.get("page") or c.get("page_start") or "?"
                doc = c.get("doc_name") or "?"
            score = float(c.get("_score") or 0.0)
            kind = (
                "web" if c.get("is_web") else
                "memory" if (c.get("is_memory") or (c.get("doc_name", "") in {"Conversation memory", "Long‑term memory"})) else
                "doc"
            )
            txt = (c.get("text") or "").strip()
            snip = txt[:220].replace("\n", " ") if txt else ""
            url = c.get("web_url") if kind == "web" else None
            viewer = False
            try:
                if (kind == "doc") and user_id and doc:
                    base = pathlib.Path("artifacts") / "docs" / str(user_id) / pathlib.Path(str(doc)).name
                    viewer = base.exists() and str(base.suffix or '').lower() == '.pdf'
            except Exception:
                viewer = False
            needle = (txt or "").strip().replace("\n", " ")[:160]
            return {
                "doc": doc,
                "page": page,
                "score": round(score, 4),
                "kind": kind,
                "snippet": snip,
                "url": url,
                "viewer": viewer,
                "needle": needle,
            }

        rr_name = str(os.environ.get("ASK_RERANKER", "off") or "off").lower()
        rr_applied = rr_name not in {"off", "none", ""}
        trace: Dict[str, Any] = {
            "only_doc": only_doc,
            "strict_docs": bool(strict_docs),
            "web_enabled": bool(web_enabled),
            "agents_enabled": bool(agents_enabled),
            "reranker": rr_name,
            "rerank_applied": bool(rr_applied),
            "retrieval": [
                {
                    "doc": (ch.get("doc_name") or "?"),
                    "page": (ch.get("page_start") or ch.get("page") or "?"),
                    "score": round(float(sc or 0.0), 4),
                    "kind": ("web" if (getattr(ch, 'get', None) and ch.get('is_web')) else "doc"),
                }
                for sc, ch in (hits[:10] or [])
            ],
            "selected_context": [_brief(c) for c in (top_chunks[:10] if top_chunks else [])],
            "web_sources": [c.get("web_url") for c in (web_chunks[:8] if web_chunks else []) if c.get("web_url")],
        }
        # Simple compare lists to visualize any ordering change
        try:
            trace["retrieval_head"] = [f"{(x.get('doc') or '?')} p.{(x.get('page') or '?')}" for x in trace.get("retrieval", [])[:5]]
            trace["selected_head"] = [f"{(x.get('doc') or '?')} p.{(x.get('page') or '?')}" for x in trace.get("selected_context", [])[:5]]
        except Exception:
            pass
        # Compute movement deltas for first N items
        try:
            def _key(d):
                return f"{d.get('doc') or '?'}|{d.get('page') or '?'}"
            old = { _key(x): i for i, x in enumerate(trace.get('retrieval', [])[:10]) }
            new = { _key(x): i for i, x in enumerate(trace.get('selected_context', [])[:10]) }
            moves = []
            for k, i_old in old.items():
                if k in new:
                    i_new = new[k]
                    delta = int(i_old) - int(i_new)
                    if delta != 0:
                        moves.append({"key": k, "old": int(i_old), "new": int(i_new), "delta": delta})
            moves.sort(key=lambda m: -abs(m.get('delta',0)))
            trace['rerank_moves'] = moves
        except Exception:
            pass
    except Exception:
        trace = {}

    if isinstance(out, dict):
        # Attach trace if available and include evidence quotes in trace
        try:
            if trace:
                # Add quotes (evidence) into the trace for UI display
                quotes = out.get("quotes") if isinstance(out.get("quotes"), list) else []
                if quotes:
                    trace["quotes"] = quotes
                # Citation validator: check overlap with selected context
                try:
                    checks = []
                    ans = (out.get('answer') or '')
                    def _parse_cite(tag):
                        import re
                        m = re.match(r"^\[([^]]+?)\s+p\.(.+)\]$", tag or "")
                        if not m: return None, []
                        doc = m.group(1); pages = []
                        for part in str(m.group(2)).split(','):
                            part = part.strip().replace('–','-')
                            if '-' in part:
                                a,b = part.split('-',1)
                                try:
                                    a=int(a); b=int(b)
                                    pages.extend(list(range(min(a,b), max(a,b)+1)))
                                except Exception: pass
                            else:
                                try: pages.append(int(part))
                                except Exception: pass
                        return doc, pages
                    ctx = trace.get('selected_context') or []
                    for tag in out.get('citations') or []:
                        doc, pages = _parse_cite(tag)
                        if not doc or not pages:
                            checks.append({'tag': tag, 'ok': False, 'reason': 'unparsed'})
                            continue
                        ok = False
                        for c in ctx:
                            if (c.get('doc') == doc) and (str(c.get('page')) in {str(p) for p in pages}):
                                try:
                                    ov = _token_overlap(ans, c.get('snippet') or '')
                                    if ov >= 0.15: ok = True; break
                                except Exception:
                                    pass
                        checks.append({'tag': tag, 'ok': bool(ok)})
                    trace['citation_check'] = checks
                except Exception:
                    pass
                out.setdefault("trace", trace)
        except Exception:
            pass

    if isinstance(out, dict):
        try:
            ans_low = (out.get('answer') or '').lower()
            nf = ('not found in sources' in ans_low) or ("couldn’t find relevant" in ans_low) or ("couldn't find relevant" in ans_low)
            meta = out.get('meta') if isinstance(out.get('meta'), dict) else {}
            if nf:
                meta['not_found'] = True
                if only_doc:
                    meta['suggest_broaden'] = True
                if not web_enabled:
                    meta['suggest_web'] = True
            if meta:
                out['meta'] = meta
        except Exception:
            pass

    if isinstance(out, dict) and web_enabled and web_chunks:
        try:
            rep = out.get("agent_report") or ""
            if rep and "Web evidence" not in rep:
                links: List[str] = []
                for ch in web_chunks[:3]:
                    url = ch.get("web_url")
                    if url:
                        links.append(f"- {url}")
                if links:
                    extra = "\n\n**Web sources**\n" + "\n".join(links)
                    out["answer"] = (out.get("answer") or "") + extra
        except Exception:
            pass

    # Attach module-level actions and next-step extraction. Merge LLM-generated
    # actions (when available) with heuristic actions so the UI always has
    # something to render. Also attach a `next_steps` array extracted from
    # the answer text.
    try:
        meta = out.get('meta') if isinstance(out.get('meta'), dict) else {}
        llm_actions: List[Dict[str, Any]] = []
        heur_actions: List[Dict[str, Any]] = []
        answer_text = out.get('answer') if isinstance(out, dict) else str(out)
        try:
            heur_actions = build_actions(answer_text or '', meta, web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=bool(strict_docs), attach_chunks=attach_chunks)
        except Exception:
            heur_actions = []

        if has_openai:
            try:
                llm_actions = generate_actions_with_llm(_orig_user_q or question, answer_text or '', trace or {}, meta or {}, web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=bool(strict_docs), attach_chunks=attach_chunks)
                if not isinstance(llm_actions, list):
                    llm_actions = []
            except Exception:
                llm_actions = []

        # Merge heuristics and LLM picks. Prefer LLM-specified fields where
        # present (label/description/recommended/available/params).
        merged: Dict[str, Dict[str, Any]] = {}
        for a in heur_actions or []:
            try:
                merged[a.get('id')] = dict(a)
            except Exception:
                continue
        for a in llm_actions or []:
            try:
                aid = a.get('id')
                if not aid:
                    continue
                if aid in merged:
                    # overwrite/augment with LLM-provided values
                    merged[aid].update({k: v for k, v in a.items() if v is not None})
                else:
                    merged[aid] = dict(a)
            except Exception:
                continue

        final_actions = list(merged.values())
        if not final_actions:
            final_actions = heur_actions or []
        if isinstance(out, dict):
            # Always assign to ensure we don't leave a None value in place.
            out['actions'] = final_actions
        else:
            # For non-dict outputs, build a minimal dict to return with actions
            out = {"answer": str(out), "citations": [], "quotes": [], "actions": final_actions}
    except Exception:
        # Last-resort: ensure actions key exists
        try:
            if isinstance(out, dict):
                out.setdefault('actions', [])
            else:
                out = {"answer": str(out), "citations": [], "quotes": [], "actions": []}
        except Exception:
            out = {"answer": str(out), "citations": [], "quotes": [], "actions": []}

    # Ensure next_steps array is attached for UI convenience
    try:
        answer_text = out.get('answer') if isinstance(out, dict) else str(out)
        # Try LLM-based next steps generation first, fallback to regex extraction
        ns = []
        if has_openai:
            try:
                ns = generate_next_steps_with_llm(_orig_user_q or question, answer_text or '', web_enabled=web_enabled, has_openai=has_openai, only_doc=only_doc, strict_docs=bool(strict_docs), attach_chunks=attach_chunks, trace=trace or {}, meta=meta or {})
                print(f"[DEBUG] LLM generated next_steps: {ns}")
            except Exception as e:
                print(f"[DEBUG] LLM next_steps failed: {e}")
                ns = extract_next_steps(answer_text or '')
        else:
            ns = extract_next_steps(answer_text or '')
            print(f"[DEBUG] Regex extracted next_steps: {ns}")
        
        # Ensure we always have some next steps (75-80% target)
        if not ns or len(ns) < 2:
            print(f"[DEBUG] Not enough next_steps ({len(ns)}), adding defaults")
            # Generate default action IDs based on context
            default_actions = []
            if not web_enabled and has_openai:
                default_actions.append("enable_web")
            if only_doc or strict_docs:
                default_actions.append("broaden_docs")
            if not attach_chunks:
                default_actions.append("upload_docs")
            default_actions.extend([
                "expand_detail",
                "generate_formula_sheet"
            ])
            print(f"[DEBUG] Default actions to consider: {default_actions}")
            # Merge with existing steps, avoiding duplicates
            for action in default_actions:
                if action not in ns:
                    ns.append(action)
                    if len(ns) >= 4:  # Limit to 4 steps max
                        break
        
        print(f"[DEBUG] Final next_steps to be added to result: {ns}")
        if isinstance(out, dict):
            out['next_steps'] = ns
    except Exception as e:
        print(f"[DEBUG] Exception in next_steps generation: {e}")
        if isinstance(out, dict):
            # Provide fallback next steps even on error
            fallback_steps = ["enable_web", "upload_docs", "expand_detail", "generate_formula_sheet"]
            out['next_steps'] = fallback_steps
            print(f"[DEBUG] Using fallback next_steps: {fallback_steps}")

    return out
