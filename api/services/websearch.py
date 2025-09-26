import json
import urllib.parse
import urllib.request
from typing import Dict, List

DEFAULT_TIMEOUT = 8
USER_AGENT = "anagnosis-bot/0.1"


def _read_json(url: str, params: Dict[str, str], timeout: int) -> Dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)

def _duckduckgo_search(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "t": "anagnosis",
    }
    try:
        data = _read_json("https://api.duckduckgo.com/", params, timeout)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    abstract = data.get("AbstractText")
    if abstract and data.get("AbstractURL"):
        out.append({
            "title": data.get("Heading") or query,
            "snippet": abstract,
            "url": data.get("AbstractURL"),
            "source": "DuckDuckGo",
        })
    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and topic.get("FirstURL"):
            out.append({
                "title": topic.get("Text") or query,
                "snippet": topic.get("Text") or "",
                "url": topic.get("FirstURL"),
                "source": "DuckDuckGo",
            })
            if len(out) >= max_results:
                break
        if isinstance(topic, dict) and topic.get("Topics"):
            for sub in topic.get("Topics"):
                if sub.get("FirstURL"):
                    out.append({
                        "title": sub.get("Text") or query,
                        "snippet": sub.get("Text") or "",
                        "url": sub.get("FirstURL"),
                        "source": "DuckDuckGo",
                    })
                    if len(out) >= max_results:
                        break
    return out[:max_results]


def search_web(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []
    return _duckduckgo_search(query, max_results, timeout)
