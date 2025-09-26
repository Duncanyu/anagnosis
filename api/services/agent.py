from typing import List, Dict, Any, Tuple
import os, re, time

try:
    from api.services import websearch
except Exception:
    websearch = None

try:
    import sympy as sp
except Exception:
    sp = None


def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower()) if s else []


def plan_queries(question: str, context_hint: str = "", k: int = 3) -> List[str]:
    if not question:
        return []
    base = question.strip()
    variants = [
        base,
        base + " key formulas",
        base + " canonical definitions",
        base + " symbols and variables list",
        "core formulas for " + base,
    ]
    out = []
    for i in range(k):
        out.append(variants[i % len(variants)])
    return list(dict.fromkeys(out))


def cluster_hits(hits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits or []:
        doc = h.get("doc") or h.get("source") or "unknown"
        ch = h.get("chapter") or h.get("section") or ""
        span = h.get("span") or f"{h.get('page','')}-{h.get('page','')}"
        key = f"{doc}|{ch}|{span}"
        clusters.setdefault(key, []).append(h)
    return clusters


def judge_answer_text(draft: str, citations: List[str]) -> float:
    if not draft:
        return 0.0
    base = 0.3
    if citations:
        base = 0.6
        if all(c in draft for c in citations):
            base = 0.9
    toks = set(_tokenize(draft))
    if any(w in toks for w in {"prove", "exercise", "show", "question"}):
        base -= 0.1
    return max(0.0, min(1.0, base))


def judge_answer(draft: str, citations: List[str]) -> Dict[str, Any]:
    s = judge_answer_text(draft, citations)
    return {"score": s, "notes": ["judge"]}


def formula_validator(formulas: List[str]) -> Dict[str, Any]:
    if not formulas:
        return {"passed": False, "notes": ["no formulas"]}
    if sp is None:
        return {"passed": True, "notes": ["sympy_unavailable"]}
    ok = True
    checked = 0
    for f in formulas:
        if checked >= 5:
            break
        try:
            if "=" in f:
                lhs, rhs = f.split("=", 1)
                eq = sp.simplify(sp.sympify(lhs) - sp.sympify(rhs))
                if eq != 0:
                    ok = False
                    break
            else:
                sp.sympify(f)
            checked += 1
        except Exception:
            ok = False
            break
    return {"passed": ok, "notes": ["validator"]}


def citation_guard(draft: str, citations: List[str], hits: List[Dict[str, Any]]) -> Tuple[str, List[str], Dict[str, Any]]:
    if not citations:
        return draft, [], {"guard": "no_citations"}
    hit_cites = set()
    for h in hits or []:
        doc = h.get("doc") or h.get("source") or ""
        pg = h.get("page")
        if doc and pg is not None:
            hit_cites.add(f"[{doc} p.{pg}]")
    filt = [c for c in citations if (c in draft) or (c in hit_cites)]
    return draft, list(dict.fromkeys(filt)), {"guard": "filtered"}


def retrieval_judge_scores(hits: List[Dict[str, Any]], question: str) -> List[Tuple[float, Dict[str, Any]]]:
    qtoks = set(_tokenize(question))
    out: List[Tuple[float, Dict[str, Any]]] = []
    for h in hits or []:
        txt = h.get("text") or ""
        htoks = set(_tokenize(txt))
        inter = len(qtoks & htoks)
        denom = max(1, len(qtoks) + len(htoks) - inter)
        j = inter / denom
        penalty = 0.0
        if any(w in txt.lower() for w in ["exercise", "show that", "prove that", "find the value"]):
            penalty = 0.1
        score = max(0.0, min(1.0, j - penalty))
        out.append((score, h))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def chapter_gatekeeper_map(hits: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    m: Dict[str, Tuple[int, int]] = {}
    for h in hits or []:
        doc = h.get("doc") or h.get("source") or "unknown"
        pg = int(h.get("page") or 0)
        ch = str(h.get("chapter") or "unknown")
        key = f"{doc}|{ch}"
        a, b = m.get(key, (pg, pg))
        m[key] = (min(a, pg), max(b, pg))
    return m


def formula_splitter(draft: str) -> str:
    if not draft:
        return draft
    keep: List[str] = []
    lines = draft.splitlines()
    pat_math = re.compile(r"[∀∃ΣΠ∑∏√≈≃≅≡≤≥±×÷∫∮∞→←↔αβγδεζηθικλμνξοπρστυφχψωA-Za-z0-9_]+\s*[=≡≈]+\s*[∀∃ΣΠ∑∏√≈≃≅≡≤≥±×÷∫∮∞→←↔αβγδεζηθικλμνξοπρστυφχψωA-Za-z0-9_]+")
    ban = re.compile(r"\b(exercise|prove|show\s+that|find\s+the\s+value|question)\b", re.I)
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if ban.search(s):
            continue
        if pat_math.search(s) or s.endswith(":") or s.count("=") >= 1:
            keep.append(ln)
    return "\n".join(keep) if keep else draft


def self_consistency_select(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cands:
        return {}
    def cite_key(x: Dict[str, Any]) -> Tuple[str, ...]:
        c = x.get("citations") or []
        return tuple(sorted(set(c)))
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for x in cands:
        buckets.setdefault(cite_key(x), []).append(x)
    best = max(buckets.values(), key=len)
    best.sort(key=lambda z: z.get("agent", {}).get("score", 0.0), reverse=True)
    return best[0]


def _sentences(s: str) -> List[str]:
    s = (s or "").replace("\n", " ").strip()
    if not s:
        return []
    parts = re.split(r"(?<=[\.!?])\s+", s)
    out: List[str] = []
    for p in parts:
        q = p.strip()
        if q:
            out.append(q)
    return out


def _chunk_texts(hits: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for h in hits or []:
        src = (h.get("doc") or h.get("source") or "").strip()
        pg = h.get("page")
        tag = f"[{src} p.{pg}]" if src and pg is not None else src
        txt = h.get("text") or ""
        if txt:
            out.append((tag, txt))
    return out


def _support_score(sent: str, cand: List[Tuple[str, str]]) -> Tuple[float, str]:
    sw = set(_tokenize(sent))
    if not sw:
        return 0.0, ""
    best = 0.0
    src = ""
    for tag, txt in cand:
        tw = set(_tokenize(txt))
        if not tw:
            continue
        inter = len(sw & tw)
        denom = max(1, len(sw) + len(tw) - inter)
        s = inter / denom
        if s > best:
            best = s
            src = tag
    return best, src


def _is_exercise_like(s: str) -> bool:
    return bool(re.search(r"\b(exercise|prove|show\s+that|determine|find\s+the\s+value|question)\b", s, re.I))


def _is_formulaish(s: str) -> bool:
    if re.search(r"[=≤≥<>±×÷∑∏∫√∂∞∇≈≃≅≡→↦↔∝∈∉∩∪⊂⊆⊇∧∨¬∀∃°]+", s):
        return True
    if re.search(r"[A-Za-z]\s*\([A-Za-z0-9, ]+\)\s*=", s):
        return True
    if re.search(r"\b\d+\/\d+\b", s):
        return True
    return False


def _sft_mask(lines: List[str]) -> List[bool]:
    try:
        from api.services.formula_cls import classify_texts as _cls
    except Exception:
        _cls = None
    if not lines:
        return []
    if _cls is None:
        return [True for _ in lines]
    try:
        labs = _cls(lines, batch_size=32, max_input_tokens=512)
        return [str(l).upper().startswith("FORMULA") for l in labs]
    except Exception:
        return [True for _ in lines]


def _trim_sentence(sent: str, limit: int = 140) -> str:
    s = (sent or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _status_from_score(score: float, threshold: float, strong: float) -> str:
    if score >= strong:
        return "supported"
    if score >= threshold:
        return "weak"
    return "missing"


def _support_badge(status: str) -> str:
    if status == "supported":
        return "✅"
    if status in {"formula", "weak"}:
        return "⚠️"
    if status == "skipped":
        return "➖"
    return "❌"


def verify_answer(question: str, out: Any, hits: List[Dict[str, Any]], time_budget_sec: int = 10, **kwargs) -> Any:
    t0 = time.time()
    policy = os.environ.get("AGENT_POLICY", "off").lower()
    enabled = as_bool(os.environ.get("ASK_AGENTS", "false")) or policy in {"light", "strong"}
    if not enabled:
        if isinstance(out, dict):
            out.setdefault("answer", "")
            out.setdefault("citations", [])
            out.setdefault("quotes", [])
            out.setdefault("agent", {})
            return out
        return {"answer": out if isinstance(out, str) else "", "citations": [], "quotes": [], "agent": {}}

    if isinstance(out, list) and out and all(isinstance(c, dict) for c in out):
        out = self_consistency_select(out)

    draft = out if isinstance(out, str) else (out.get("answer") or out.get("text") or out.get("draft") or "")
    citations = [] if isinstance(out, str) else list(out.get("citations", []))
    diag: Dict[str, Any] = {}
    diag["policy"] = policy
    diag["enabled"] = True

    base_cand = _chunk_texts(hits)
    sens = _sentences(draft)
    keep_sens: List[str] = []
    kept_set = set()
    supp_cites: List[str] = []
    th = 0.08
    formula_mode = as_bool(os.environ.get("ASK_FORMULA_MODE", "false"))

    sent_infos: List[Dict[str, Any]] = []
    for s in sens:
        sc, tag = _support_score(s, base_cand)
        entry = {"sentence": s, "score": sc, "source": tag or "", "status": "unchecked"}
        sent_infos.append(entry)
        if formula_mode and _is_exercise_like(s):
            entry["status"] = "skipped"
            continue
        if sc >= th or (formula_mode and _is_formulaish(s)):
            keep_sens.append(s)
            kept_set.add(s)
            if tag:
                supp_cites.append(tag)
            entry["status"] = "supported" if sc >= th else "formula"
        elif sc > 0:
            entry["status"] = "weak"

    if formula_mode and keep_sens:
        m = _sft_mask(keep_sens)
        filtered = []
        for s, flag in zip(keep_sens, m):
            if flag:
                filtered.append(s)
            else:
                for info in sent_infos:
                    if info["sentence"] == s:
                        info["status"] = "skipped"
                        break
        keep_sens = filtered
        kept_set = set(keep_sens)

    if not keep_sens and sens:
        keep_sens = sens[:1]
        kept_set = set(keep_sens)

    pruned = " ".join(keep_sens).strip()

    more_cand: List[Tuple[str, str]] = []
    if len(pruned) < max(48, int(len(draft) * 0.4)) and (time.time() - t0) < time_budget_sec * 0.7:
        try:
            from api.services.index import search as _search
            qvars = plan_queries(question, k=2)
            strict = as_bool(os.environ.get("ASK_STRICT_DOC", "false"))
            more_hits: List[Tuple[float, Dict[str, Any]]] = []
            for qv in qvars:
                rem = max(1, int(time_budget_sec - (time.time() - t0)))
                if rem <= 0:
                    break
                res = _search(qv, k=6, time_budget_sec=min(6, rem), strict_doc=strict)
                more_hits.extend(res.get("hits", []))
            more_cand = _chunk_texts([h[1] for h in more_hits])
            for info in sent_infos:
                s = info["sentence"]
                if s in kept_set or info["status"] == "skipped":
                    continue
                sc2, tag2 = _support_score(s, more_cand)
                if sc2 > info["score"]:
                    info["score"] = sc2
                    info["source"] = tag2 or info["source"]
                if sc2 >= th or (formula_mode and _is_formulaish(s)):
                    keep_sens.append(s)
                    kept_set.add(s)
                    if tag2:
                        supp_cites.append(tag2)
                    info["status"] = "supported" if sc2 >= th else "formula"
            pruned = " ".join(keep_sens).strip()
        except Exception:
            pass

    supp_cites.extend(citations)
    cites = []
    seen = set()
    for c in supp_cites:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        cites.append(c)

    def extract_quotes_from_hits(hh: List[Tuple[str, str]], max_quotes: int = 5) -> List[Dict[str, str]]:
        out_q: List[Dict[str, str]] = []
        for tag, txt in hh:
            q = (txt or "").strip()
            if not q:
                continue
            if len(q) > 220:
                cut = q[:220]
                if "." in cut:
                    cut = cut.rsplit(".", 1)[0] + "."
                q = cut
            out_q.append({"quote": q, "source": tag})
            if len(out_q) >= max_quotes:
                break
        return out_q

    quotes = extract_quotes_from_hits(base_cand)

    diag["changed"] = bool(pruned and pruned != draft)

    strong_th = 0.2
    for info in sent_infos:
        if info["status"] in {"supported", "formula", "skipped"}:
            continue
        info["status"] = _status_from_score(info["score"], th, strong_th)

    diag["kept_sentences"] = len(keep_sens)
    diag["pruned"] = len(sens) - len(keep_sens)
    diag["total_sentences"] = len(sens)
    diag["verdict"] = "trimmed" if len(keep_sens) < len(sens) else "validated"
    diag["time_sec"] = round(time.time() - t0, 3)

    support_rows = []
    for info in sent_infos:
        trimmed = _trim_sentence(info["sentence"], limit=160).replace("|", "\\|")
        support_rows.append({
            "sentence": trimmed,
            "score": round(float(info["score"]), 3),
            "source": info["source"] or "-",
            "status": info["status"],
        })

    diag["support"] = support_rows
    status_counts: Dict[str, int] = {}
    for row in support_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    diag["status_counts"] = status_counts

    web_results: List[Dict[str, Any]] = []
    if websearch is not None and as_bool(os.environ.get("ASK_WEB_SEARCH", "false")):
        unresolved = [info for info in sent_infos if info["status"] in {"weak", "missing"}]
        queries: List[str] = []
        queries.extend([info["sentence"] for info in unresolved[:2]])
        if not queries:
            queries.append(question)
        seen = set()
        for qv in queries:
            res = websearch.search_web(qv, max_results=4)
            if not res:
                continue
            for item in res:
                url = item.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                web_results.append(item)
            if len(web_results) >= 6:
                break
    diag["web_results"] = web_results

    report_lines = ["| Sentence | Support | Source |", "| --- | --- | --- |"]
    for row in support_rows[:12]:
        badge = _support_badge(row["status"])
        report_lines.append(f"| {row['sentence']} | {row['score']:.2f} {badge} | {row['source']} |")
    report_md = "\n".join(report_lines)
    diag["report_md"] = report_md

    if web_results:
        web_lines = ["| Result | Snippet |", "| --- | --- |"]
        for item in web_results[:6]:
            title = item.get("title") or item.get("url") or "result"
            url = item.get("url")
            snippet = (item.get("snippet") or "").replace("|", " ")
            title_md = f"[{title}]({url})" if url else title
            web_lines.append(f"| {title_md} | {snippet} |")
        diag["web_results_md"] = "\n".join(web_lines)

    if isinstance(out, dict):
        out_ans = pruned if pruned else draft
        out["answer"] = out_ans
        out["citations"] = cites
        out["quotes"] = quotes
        agent_state = dict(out.get("agent", {}))
        agent_state.update(diag)
        out["agent"] = agent_state
        out["agent_meta"] = diag
        out["agent_report"] = report_md
        return out

    return {
        "answer": pruned if pruned else draft,
        "citations": cites,
        "quotes": quotes,
        "agent": diag,
        "agent_meta": diag,
        "agent_report": report_md,
    }
