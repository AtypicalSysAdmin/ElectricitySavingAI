#!/usr/bin/env python3
"""
ask.py - Fast Interactive Ollama CLI with web search
Usage: python ask.py [-m model] [-s]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse

OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:latest"

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
    "Connection": "keep-alive",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_blocked(url: str) -> bool:
    try:
        host = re.sub(r"^www\.", "", urllib.parse.urlparse(url).netloc.lower())
        return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)
    except Exception:
        return False

def fetch_local_models() -> list:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

CODE_TOKENS = {"python", "bash", "shell", "script", "code", "function", "command",
               "snippet", "oneliner", "one-liner", "regex", "sql", "js", "javascript",
               "typescript", "rust", "go", "c++", "java", "how to", "example"}
FACT_TOKENS = {"what is", "who is", "when did", "where is", "define", "meaning of",
               "version", "latest", "current", "price", "release"}

def classify_query(q: str) -> str:
    ql = q.lower()
    if any(t in ql for t in CODE_TOKENS): return "code"
    if any(t in ql for t in FACT_TOKENS): return "fact"
    return "general"

# ---------------------------------------------------------------------------
# Follow-up resolver
# ---------------------------------------------------------------------------

REFERENTIAL = {"that", "it", "this", "those", "them", "there", "the same",
               "more", "more about", "what about", "and", "also", "latest on"}

def resolve_query(query: str, history: list) -> str:
    if not history:
        return query
    ql = query.lower().strip()
    is_vague = (
        len(ql.split()) <= 6
        or any(ql.startswith(r) or ql == r for r in REFERENTIAL)
        or bool(re.search(r"\b(it|that|this|those|them)\b", ql))
    )
    if not is_vague:
        return query
    last_user = next((e[5:].strip() for e in reversed(history) if e.startswith("User:")), "")
    if not last_user:
        return query
    stop = {"what","is","the","a","an","how","why","when","where","who",
            "do","does","did","can","could","tell","me","about","give","some"}
    words = [w for w in re.sub(r"[^\w\s]", "", last_user.lower()).split() if w not in stop]
    topic = " ".join(words[:6])
    if not topic:
        return query
    resolved = f"{query} {topic}"
    print(f"🔗 Resolved: \"{resolved}\"", file=sys.stderr)
    return resolved

# ---------------------------------------------------------------------------
# Web search — snippets only, fast, many sources
# ---------------------------------------------------------------------------

def web_search(query: str) -> tuple[str, str]:
    try:
        from ddgs import DDGS
        # Fetch more raw results to survive blocked-domain filtering
        raw = DDGS().text(query, max_results=12)
        if not raw:
            return "No results found.", ""

        results = [r for r in raw if not is_blocked(r.get("href", ""))]
        if not results:
            return "No unblocked results found.", ""

        # Score each snippet by keyword overlap — surfaces most relevant first
        qwords = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        scored = []
        for r in results:
            body  = r.get("body",  "").strip()
            title = r.get("title", "").strip()
            href  = r.get("href",  "")
            if not body:
                continue
            score = sum(1 for w in qwords if w in body.lower())
            scored.append((score, title, body, href))

        scored.sort(key=lambda x: x[0], reverse=True)

        snippets = [
            f"• [{title}] {body}  <{href}>"
            for _, title, body, href in scored[:8]   # up to 8 diverse sources
        ]

        display = "\n".join(snippets[:5])
        context = "## Search Snippets\n" + "\n".join(snippets)
        return display, context

    except Exception as e:
        return f"Search failed: {e}", ""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, context: str, history: list) -> str:
    qtype = classify_query(query)
    parts = []

    if context:
        parts.append(
            "SYSTEM: You are a precise assistant. "
            "The web search results below are your PRIMARY source of truth. "
            "Use them directly. Do not hallucinate or add unsourced facts.\n\n"
            f"{context}"
        )

    if history:
        parts.append("## Recent conversation\n" + "\n".join(history))

    if qtype == "code":
        instr = ("Return ONLY the code or command. No explanation unless asked. "
                 "Use a code block. No preamble.")
    elif qtype == "fact":
        instr = ("State the fact in one sentence. "
                 "One line of context max. No lists, no padding.")
    else:
        instr = ("Answer in 2-4 sentences max. Be direct. "
                 "No filler like 'Great question'. If unsure, say so in one sentence.")

    parts.append(f"## Question\n{query}\n\n## Instructions\n{instr}")
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ask_ollama(prompt: str, model: str, think_enabled: bool) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":    0.05,
            "num_predict":    256,
            "num_ctx":        2048,
            "top_k":          10,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
        }
    }
    if not think_enabled:
        payload["think"] = False

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
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

def print_interface(model: str, search: bool, think: bool):
    print("\n" + "="*52)
    print(f"  Model : {model}")
    print(f"  Search: {'ON  (8 sources, scored)' if search else 'OFF'}")
    print(f"  Think : {'VISIBLE' if think else 'HIDDEN'}")
    print("="*52)
    print("  /search  Toggle web search")
    print("  /think   Toggle model thinking visibility")
    print("  /model   Switch model  (e.g. /model llama3)")
    print("  /models  List local models")
    print("  /clear   Reset conversation history")
    print("  /help    Show this menu")
    print("  /exit    Quit")
    print("="*52 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",  default=DEFAULT_MODEL)
    parser.add_argument("-s", "--search", action="store_true")
    args = parser.parse_args()

    model          = args.model
    search_enabled = args.search
    think_enabled  = False
    history        = []

    print_interface(model, search_enabled, think_enabled)

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
                print("🧠 History cleared.\n")
            elif cmd == "/help":
                print_interface(model, search_enabled, think_enabled)
            elif cmd == "/search":
                search_enabled = not search_enabled
                print_interface(model, search_enabled, think_enabled)
            elif cmd == "/think":
                think_enabled = not think_enabled
                print_interface(model, search_enabled, think_enabled)
            elif cmd in ["/model", "/models"]:
                local_models = fetch_local_models()
                if not local_models:
                    print("Could not reach Ollama.\n")
                    continue
                if len(parts) > 1:
                    target   = parts[1]
                    resolved = target if target in local_models else f"{target}:latest"
                    if resolved in local_models:
                        model = resolved
                        print_interface(model, search_enabled, think_enabled)
                    else:
                        print(f"Model '{target}' not found locally.\n")
                else:
                    print("\nLocal models:")
                    for m in local_models:
                        print(f"  {'▶' if m == model else ' '} {m}")
                    print("\nUse /model [name] to switch.\n")
            else:
                print(f"Unknown command: {cmd}\n")
            continue

        # Resolve vague follow-ups, then search + respond
        search_query = resolve_query(user_input, history)
        context_str  = ""

        if search_enabled:
            print("🔍 Searching...", file=sys.stderr)
            display, context_str = web_search(search_query)
            if context_str:
                print("\n🌐 [Search — top sources]")
                print(display)
                print("-" * 52)

        full_prompt  = build_prompt(user_input, context_str, history)

        print(f"🤖 [{model}]...", file=sys.stderr, end="\r")
        t0       = time.time()
        response = ask_ollama(full_prompt, model, think_enabled)
        elapsed  = time.time() - t0
        print(" " * 40, end="\r", file=sys.stderr)

        print(f"\n{response}")
        print(f"⏱️  {elapsed:.2f}s\n")

        if "Ollama error" not in response:
            history.append(f"User: {user_input}")
            history.append(f"AI: {response}")
            if len(history) > 6:
                history = history[-6:]

if __name__ == "__main__":
    main()