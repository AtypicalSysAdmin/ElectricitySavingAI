#!/usr/bin/env python3
"""
ask.py - Minimal Interactive Ollama CLI with response benchmarking
Usage: python ask.py [-m model] [-s]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:latest"

# Domains blocked from search results and page fetching
BLOCKED_DOMAINS = {
    "wikipedia.org", "wikimedia.org", "wikidata.org", "wikihow.com",
    "wikia.com", "fandom.com", "wiki.org",
    "quora.com", "reddit.com", "answers.com", "ask.com",
    "britannica.com", "encyclopedia.com", "infoplease.com",
    "about.com", "reference.com", "thoughtco.com",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# ---------------------------------------------------------------------------
# Query classification — drives prompt style and search depth
# ---------------------------------------------------------------------------

CODE_TOKENS = {"python", "bash", "shell", "script", "code", "function", "command",
               "snippet", "oneliner", "one-liner", "regex", "sql", "js", "javascript",
               "typescript", "rust", "go", "c++", "java", "how to", "example"}

FACT_TOKENS = {"what is", "who is", "when did", "where is", "define", "meaning of",
               "version", "latest", "current", "price", "release"}

def classify_query(q: str) -> str:
    ql = q.lower()
    if any(t in ql for t in CODE_TOKENS):
        return "code"
    if any(t in ql for t in FACT_TOKENS):
        return "fact"
    return "general"

# ---------------------------------------------------------------------------
# Page fetching — target semantic content tags, skip boilerplate
# ---------------------------------------------------------------------------

def _extract_text(html: str, query_words: set, max_chars: int = 1500) -> str:
    # Remove noise blocks entirely
    html = re.sub(
        r"<(script|style|nav|footer|header|form|aside|iframe|noscript|svg)[^>]*>.*?</\1>",
        "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Prefer semantic content zones
    for tag in ("article", "main", "section", "div"):
        block = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, flags=re.DOTALL | re.IGNORECASE)
        if block:
            candidate = re.sub(r"<[^>]+>", " ", block.group(1))
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) > 300:
                html = block.group(1)
                break

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    # Score sentences by keyword overlap with the query
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 40 or len(s) > 600:
            continue
        sl = s.lower()
        score = sum(1 for w in query_words if w in sl)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored[:12]]
    return " ".join(selected)[:max_chars]

def is_blocked(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
        host = re.sub(r"^www\.", "", host)
        return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)
    except Exception:
        return False

def fetch_page(url: str, query_words: set) -> str:
    if is_blocked(url):
        return ""
    try:
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=4) as r:
            raw = r.read()
            try:
                import gzip
                raw = gzip.decompress(raw)
            except Exception:
                pass
            html = raw.decode("utf-8", errors="ignore")
        return _extract_text(html, query_words)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Query resolver -- expand vague follow-ups using conversation history
# ---------------------------------------------------------------------------

REFERENTIAL = {"that", "it", "this", "those", "them", "there", "the same",
               "more", "more about", "what about", "and", "also", "latest on"}

def resolve_query(query: str, history: list) -> str:
    """If query is a vague follow-up, inject the last topic from history."""
    if not history:
        return query

    ql = query.lower().strip()

    is_vague = (
        len(ql.split()) <= 6
        or any(ql.startswith(r) or ql == r for r in REFERENTIAL)
        or re.search(r"\b(it|that|this|those|them)\b", ql)
    )

    if not is_vague:
        return query

    last_user = ""
    for entry in reversed(history):
        if entry.startswith("User:"):
            last_user = entry[5:].strip()
            break

    if not last_user:
        return query

    stop = {"what","is","the","a","an","how","why","when","where","who",
            "do","does","did","can","could","tell","me","about","give","some"}
    words = [w for w in re.sub(r"[^\w\s]", "", last_user.lower()).split() if w not in stop]
    topic = " ".join(words[:6])

    if not topic:
        return query

    resolved = f"{query} {topic}"
    print(f"\U0001f517 Resolved query: \"{resolved}\"", file=__import__("sys").stderr)
    return resolved

# ---------------------------------------------------------------------------
# Web search — parallel page fetching for top results
# ---------------------------------------------------------------------------

def web_search(query: str, deep: bool = False) -> tuple[str, str]:
    """Returns (display_text, full_context_for_model)."""
    try:
        from ddgs import DDGS

        raw_results = DDGS().text(query, max_results=7)
        if not raw_results:
            return "No results found.", ""

        results = [r for r in raw_results if not is_blocked(r.get("href", ""))]
        if not results:
            return "No unblocked results found.", ""

        query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())

        snippets = []
        for r in results[:5]:
            title = r.get("title", "").strip()
            body  = r.get("body",  "").strip()
            href  = r.get("href",  "")
            if body:
                snippets.append(f"• [{title}] {body}  <{href}>")

        ctx_parts = []
        if snippets:
            ctx_parts.append("## Search Snippets\n" + "\n".join(snippets))

        # Only fetch full pages when /deep is active
        if deep:
            urls = [r.get("href", "") for r in results[:2] if r.get("href") and not is_blocked(r.get("href",""))]
            page_texts = {}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(fetch_page, u, query_words): u for u in urls}
                try:
                    for fut in as_completed(futures, timeout=4):
                        u = futures[fut]
                        try:
                            txt = fut.result()
                            if txt:
                                page_texts[u] = txt
                        except Exception:
                            pass
                except TimeoutError:
                    pass
            for i, url in enumerate(urls):
                if url in page_texts:
                    ctx_parts.append(f"## Page {i+1} Content  <{url}>\n{page_texts[url]}")

        display = "\n".join(snippets[:4])
        context = "\n\n".join(ctx_parts)
        return display, context

    except Exception as e:
        return f"Search failed: {e}", ""


# ---------------------------------------------------------------------------
# Prompt builder — query-type-aware instructions
# ---------------------------------------------------------------------------

def build_prompt(query: str, context: str, history: list) -> str:
    qtype = classify_query(query)
    parts = []

    if context:
        parts.append(
            "SYSTEM: You are a precise assistant. The web search results below are "
            "your PRIMARY source of truth. Use them. Do not hallucinate or add unsourced facts.\n\n"
            f"{context}"
        )

    if history:
        parts.append("## Recent conversation\n" + "\n".join(history))

    if qtype == "code":
        instruction = (
            "Return ONLY the code or command that answers the question. "
            "No explanation unless the user asked for one. "
            "Use a code block. No preamble."
        )
    elif qtype == "fact":
        instruction = (
            "State the fact in one sentence. "
            "Add one line of context only if it directly helps. "
            "No lists, no padding."
        )
    else:
        instruction = (
            "Answer in 2-4 sentences max. "
            "Be direct. No filler phrases like 'Great question' or 'Certainly'. "
            "If unsure, say so in one sentence."
        )

    parts.append(f"## Question\n{query}\n\n## Instructions\n{instruction}")
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def ask_ollama(prompt: str, model: str, think_enabled: bool) -> str:
    payload_dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": 256,
            "num_ctx": 2048,
            "top_k": 10,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
    }
    if not think_enabled:
        payload_dict["think"] = False

    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json", **HTTP_HEADERS},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        return f"Ollama error: {e}"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def print_interface(model: str, search: bool, think: bool, deep: bool = False):
    print("\n" + "="*50)
    print(f"Model: {model} | Search: {'ON' if search else 'OFF'} | Deep: {'ON' if deep else 'OFF'} | Think: {'VISIBLE' if think else 'HIDDEN'}")
    print("="*50)
    print("Commands:")
    print("  /clear   Reset chat history")
    print("  /search  Toggle web search")
    print("  /think   Toggle thinking process visibility")
    print("  /model   Change Ollama model (e.g. /model llama3)")
    print("  /help    Show this dashboard")
    print("  /exit    Exit chat (or Ctrl+C)")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Fast local AI lookup loop")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("-s", "--search", action="store_true", help="Enable search by default")
    args = parser.parse_args()

    model          = args.model
    search_enabled = args.search
    deep_enabled   = False
    think_enabled  = False
    history        = []

    print_interface(model, search_enabled, think_enabled, deep_enabled)

    while True:
        try:
            user_input = input("✨ ask > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd in ["/clear", "/forget"]:
                history = []
                print("🧠 Conversation history cleared.\n")
                continue
            elif cmd == "/help":
                print_interface(model, search_enabled, think_enabled, deep_enabled)
                continue
            elif cmd == "/search":
                search_enabled = not search_enabled
                print_interface(model, search_enabled, think_enabled, deep_enabled)
                continue
            elif cmd == "/think":
                think_enabled = not think_enabled
                print_interface(model, search_enabled, think_enabled, deep_enabled)
                continue
            elif cmd in ["/model", "/models"]:
                local_models = fetch_local_models()
                if not local_models:
                    print("Could not retrieve local models.\n")
                    continue
                if len(parts) > 1:
                    target = parts[1]
                    resolved = target if target in local_models else f"{target}:latest"
                    if resolved in local_models:
                        model = resolved
                        print_interface(model, search_enabled, think_enabled, deep_enabled)
                    else:
                        print(f"Model '{target}' not found locally.\n")
                else:
                    print("\nLocal Models:")
                    for m in local_models:
                        print(f"  {'*' if m == model else ' '} {m}")
                    print("\nType '/model [name]' to switch.\n")
                continue
            else:
                print(f"Unknown command: {cmd}\n")
                continue

        # Build context — resolve vague follow-ups before searching
        search_query = resolve_query(user_input, history)
        context_str = ""
        if search_enabled:
            print("🔍 Searching...", file=sys.stderr)
            display, context_str = web_search(search_query, deep=deep_enabled)
            if context_str:
                print("\n🌐 [Search Context]")
                print(display)
                print("-" * 50)

        full_prompt = build_prompt(user_input, context_str, history)

        print(f"🤖 [{model}] Thinking...", file=sys.stderr, end="\r")
        start_time   = time.time()
        response     = ask_ollama(full_prompt, model, think_enabled)
        elapsed_time = time.time() - start_time
        print(" " * 50, end="\r", file=sys.stderr)

        print(f"\n{response}")
        print(f"⏱️  {elapsed_time:.2f}s\n")

        if "Ollama error" not in response:
            history.append(f"User: {user_input}")
            history.append(f"AI: {response}")
            if len(history) > 6:
                history = history[-6:]

if __name__ == "__main__":
    main()