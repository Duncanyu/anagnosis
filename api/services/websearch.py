import json
import urllib.parse
import urllib.request
import os
from typing import Dict, List

from api.core.config import load_config

DEFAULT_TIMEOUT = 8
USER_AGENT = "anagnosis-bot/0.1"


def _setting(name: str, default: str = "") -> str:
    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    val = cfg.get(name)
    if isinstance(val, str) and val.strip():
        return val.strip()
    env = os.getenv(name)
    if env:
        return env
    return default


def get_provider_name() -> str:
    """Determine provider based on config and available keys.

    - If WEB_SEARCH_PROVIDER is set to 'brave'|'serpapi'|'duckduckgo'|'auto', honor it.
    - If 'auto' or empty: prefer a provider with a configured key (serpapi or brave).
      If both keys exist, default to 'brave' unless explicitly set elsewhere.
    - Fallback to 'duckduckgo'.
    """
    prov = (_setting("WEB_SEARCH_PROVIDER", "")).lower().strip()
    brave_key = _setting("BRAVE_API_KEY") or _setting("BRAVE_SEARCH_KEY")
    serp_key = _setting("SERPAPI_KEY")
    if prov in {"brave", "serpapi", "duckduckgo"}:
        return prov
    # auto / unset
    if prov == "auto" or prov == "":
        if serp_key and not brave_key:
            return "serpapi"
        if brave_key and not serp_key:
            return "brave"
        if brave_key and serp_key:
            # default preference when both available (changeable via settings)
            return "brave"
    return "duckduckgo"


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


def _serpapi_search(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    api_key = _setting("SERPAPI_KEY")
    if not api_key:
        return []
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": api_key,
    }
    try:
        data = _read_json("https://serpapi.com/search.json", params, timeout)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for item in data.get("organic_results", [])[:max_results]:
        out.append({
            "title": item.get("title") or "",
            "snippet": item.get("snippet") or " ".join(item.get("snippet_highlighted_words", [])),
            "url": item.get("link") or "",
            "source": item.get("displayed_link") or "web",
        })
    return [r for r in out if r["url"]]


def _brave_search(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    api_key = _setting("BRAVE_API_KEY") or _setting("BRAVE_SEARCH_KEY")
    if not api_key:
        return []
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        # Brave Search API expects X-Subscription-Token per docs
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": max_results,
    }
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for item in payload.get("web", {}).get("results", [])[:max_results]:
        out.append({
            "title": item.get("title") or "",
            "snippet": item.get("snippet") or "",
            "url": item.get("url") or "",
            "source": "Brave",
        })
    return [r for r in out if r["url"]]


def search_web(query: str, max_results: int = 5, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []

    provider = get_provider_name()
    if provider == "brave":
        results = _brave_search(query, max_results, timeout)
        if results:
            return results
        provider = "duckduckgo"
    if provider == "serpapi":
        results = _serpapi_search(query, max_results, timeout)
        if results:
            return results
        provider = "duckduckgo"

    results = _duckduckgo_search(query, max_results, timeout)
    if not results:
        # final fallback: try SerpAPI if key exists, then Brave
        alt = _serpapi_search(query, max_results, timeout)
        if alt:
            return alt
        alt = _brave_search(query, max_results, timeout)
        if alt:
            return alt
    return results
