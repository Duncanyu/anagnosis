from typing import List, Dict, Any

def plan_queries(question: str, context_hint: str = "", k: int = 3) -> List[str]:
    return [question] if not question else [question + f" (variant {i+1})" for i in range(k)]

def cluster_hits(hits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        key = f"{h.get('doc','unknown')}:{h.get('span','0-0')}"
        clusters.setdefault(key, []).append(h)
    return clusters

def judge_answer(draft: str, citations: List[str]) -> Dict[str, Any]:
    score = 0.0
    if citations and all(c in draft for c in citations):
        score = 0.9
    elif citations:
        score = 0.6
    else:
        score = 0.3
    return {"score": score, "notes": ["auto-judged"]}

def cas_check(formulas: List[str]) -> Dict[str, Any]:
    passed = bool(formulas)
    return {"passed": passed, "notes": ["CAS check placeholder"]}

def verify_answer(draft: str, citations: List[str], hits: List[Dict[str, Any]], timeout: int = 10) -> Dict[str, Any]:
    judgment = judge_answer(draft, citations)
    score = judgment.get("score", 0.0)
    clusters = cluster_hits(hits)
    alts = []
    if score < 0.7:
        alts = plan_queries(draft)
    notes = judgment.get("notes", []) + ["agent verification complete"]
    return {
        "draft": draft,
        "score": score,
        "clusters": clusters,
        "alts": alts,
        "notes": notes
    }