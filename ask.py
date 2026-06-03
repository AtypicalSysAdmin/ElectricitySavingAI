#!/usr/bin/env python3
"""
ask.py - Fast Interactive Ollama CLI with web search (v5)
Usage: python ask.py [-m model] [-s]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:latest"

BLOCKED_DOMAINS = {
    "wikipedia.org", "wikimedia.org", "wikidata.org", "wikihow.com",
    "wikia.com", "fandom.com", "wiki.org",
    "quora.com", "reddit.com", "answers.com", "ask.com",
    "britannica.com", "encyclopedia.com", "infoplease.com",
    "about.com", "reference.com", "thoughtco.com",
}

TRUST_TIERS: dict[str, int] = {
    "learn.microsoft.com": 5, "support.microsoft.com": 5,
    "microsoft.com": 4, "apple.com": 4, "developer.android.com": 4,
    "docs.python.org": 5, "docs.docker.com": 5, "docs.aws.amazon.com": 5,
    "cloud.google.com": 5, "docs.github.com": 5, "nodejs.org": 5,
    "dev.mysql.com": 5, "postgresql.org": 5, "man7.org": 5,
    "kernel.org": 5, "nginx.org": 5, "apache.org": 5,
    "stackoverflow.com": 4, "serverfault.com": 4, "superuser.com": 4,
    "askubuntu.com": 4, "stackexchange.com": 4,
    "github.com": 3, "gitlab.com": 3, "pypi.org": 3,
    "npmjs.com": 3, "crates.io": 3, "archlinux.org": 3,
    "digitalocean.com": 3, "linuxize.com": 3, "ss64.com": 3,
    "realpython.com": 2, "baeldung.com": 2, "geeksforgeeks.org": 2,
    "tutorialspoint.com": 2, "w3schools.com": 2, "freecodecamp.org": 2,
    "tecmint.com": 2, "howtogeek.com": 2, "thewindowsclub.com": 2,
    "operavps.com": 2, "netwrix.com": 2, "powershellfaqs.com": 2,
    "medium.com": 1, "dev.to": 1, "hashnode.dev": 1,
    "towardsdatascience.com": 1, "analyticsvidhya.com": 1,
    "codepal.ai": 0, "phind.com": 0, "blackbox.ai": 0,
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

STOP_WORDS = frozenset(
    "what is the a an how why when where who do does did can could tell me "
    "about give some to for in on of and or but with".split()
)

# ---------------------------------------------------------------------------
# HTML → plain-text (stdlib, fast)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return p.get_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_blocked(url: str) -> bool:
    try:
        host = re.sub(r"^www\.", "", urllib.parse.urlparse(url).netloc.lower())
        return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)
    except Exception:
        return False


def domain_trust(url: str) -> int:
    host = re.sub(r"^www\.", "", urllib.parse.urlparse(url).netloc.lower())
    for domain, score in TRUST_TIERS.items():
        if host == domain or host.endswith("." + domain):
            return score
    return 2


def fetch_local_models() -> list[str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Model pre-warm — load into memory once at startup
# ---------------------------------------------------------------------------

def prewarm_model(model: str) -> None:
    """Send a tiny request so Ollama loads the model into GPU/RAM."""
    payload = {"model": model, "prompt": "hi", "stream": False,
               "options": {"num_predict": 1, "num_ctx": 32},
               "keep_alive": "30m"}
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
        print("  ⚡ Model loaded & warm", file=sys.stderr)
    except Exception:
        print("  ⚠️  Pre-warm failed (first query may be slow)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Page fetcher — parallel
# ---------------------------------------------------------------------------

def fetch_page(url: str, byte_limit: int = 15_000) -> str:
    try:
        if is_blocked(url):
            return ""
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=4) as r:
            raw = r.read(byte_limit).decode("utf-8", errors="ignore")
        return html_to_text(raw)
    except Exception:
        return ""


def fetch_pages_parallel(urls: list[str], max_chars: int = 2500,
                         workers: int = 4) -> dict[str, str]:
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_page, u): u for u in urls}
        for fut in as_completed(futures, timeout=6):
            url = futures[fut]
            try:
                text = fut.result()
                if text:
                    results[url] = text[:max_chars]
            except Exception:
                pass
    return results


# ---------------------------------------------------------------------------
# Query complexity
# ---------------------------------------------------------------------------

def query_complexity(query: str) -> str:
    ql = query.lower()
    words = ql.split()
    content = [w for w in re.sub(r"[^\w\s]", "", ql).split() if w not in STOP_WORDS]

    if len(words) > 18 or any(t in ql for t in (
        "compare", "difference between", "explain in detail", "step by step",
        "pros and cons", "trade-off", "versus", " vs ",
    )):
        return "complex"
    if len(content) <= 4 and len(words) <= 10:
        return "simple"
    return "medium"


# Adaptive profiles — simple is aggressively lean
COMPLEXITY_PROFILE = {
    #                  predict  ctx    snip  ddgs  pages  pgchars  fetch_pages
    "simple":  dict(num_predict=192,  num_ctx=1536, snippets=3,
                    search_results=5, pages=0, page_chars=0),
    "medium":  dict(num_predict=384,  num_ctx=3072, snippets=5,
                    search_results=8, pages=2, page_chars=2000),
    "complex": dict(num_predict=1024, num_ctx=4096, snippets=8,
                    search_results=12, pages=4, page_chars=3000),
}


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

EXPLICIT_CODE = {
    "python", "bash", "shell", "script", "code", "function", "snippet",
    "oneliner", "one-liner", "regex", "sql", "js", "javascript",
    "typescript", "rust", "go", "c++", "java",
}
HOWTO_TOKENS = {"how to", "how do i", "how can i", "steps to", "way to", "command"}
FACT_TOKENS  = {
    "what is", "who is", "when did", "where is", "define", "meaning of",
    "version", "latest", "current", "price", "release date",
}


def classify_query(q: str) -> str:
    ql = q.lower()
    words = set(ql.split())
    if words & EXPLICIT_CODE and not any(t in ql for t in HOWTO_TOKENS):
        return "code"
    if any(t in ql for t in HOWTO_TOKENS):
        return "howto"
    if any(t in ql for t in FACT_TOKENS):
        return "fact"
    return "general"


# ---------------------------------------------------------------------------
# Follow-up resolver
# ---------------------------------------------------------------------------

REFERENTIAL_STARTS = {
    "that", "it", "this", "those", "them", "there", "the same",
    "more", "more about", "what about", "and also", "also",
    "latest on", "why", "elaborate", "explain",
}


def _content_words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s]", "", text.lower()).split() if w not in STOP_WORDS]


def resolve_query(query: str, history: list[str]) -> str:
    if not history:
        return query
    ql = query.lower().strip()
    explicitly_referential = any(ql.startswith(r) for r in REFERENTIAL_STARTS)
    pronoun_only = bool(re.match(r"^(it|that|this|those|them)\b", ql))
    if len(_content_words(query)) >= 2 and not explicitly_referential and not pronoun_only:
        return query
    last_user = next((e[5:].strip() for e in reversed(history) if e.startswith("User:")), "")
    if not last_user:
        return query
    topic = " ".join(_content_words(last_user)[:5])
    if not topic:
        return query
    resolved = f"{query} {topic}"
    print(f"🔗 Resolved: \"{resolved}\"", file=sys.stderr)
    return resolved


# ---------------------------------------------------------------------------
# Web search — adaptive: snippets-only for simple, pages for medium+
# ---------------------------------------------------------------------------

def _keyword_score(query_words: set[str], title: str, body: str) -> float:
    tl, bl = title.lower(), body.lower()
    return sum(2 for w in query_words if w in tl) + sum(1 for w in query_words if w in bl)


def web_search(query: str, profile: dict) -> tuple[str, str]:
    try:
        from ddgs import DDGS                                    # pip install ddgs
        raw = DDGS().text(query, max_results=profile["search_results"])
        if not raw:
            return "No results found.", ""

        results = [r for r in raw if not is_blocked(r.get("href", ""))]
        if not results:
            return "No unblocked results found.", ""

        qwords = set(re.sub(r"[^\w\s]", "", query.lower()).split()) - STOP_WORDS
        scored: list[tuple[float, dict]] = []
        for r in results:
            body  = r.get("body",  "").strip()
            title = r.get("title", "").strip()
            href  = r.get("href",  "")
            if not body:
                continue
            trust   = domain_trust(href)
            keyword = _keyword_score(qwords, title, body)
            scored.append((trust * 3.0 + keyword, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:profile["snippets"]]

        # --- Display ---
        display_lines = []
        for _, r in top[:5]:
            trust = domain_trust(r["href"])
            star  = "★" * min(trust, 5)
            display_lines.append(
                f"• [{r.get('title','')}] {star}  "
                f"{r.get('body','')[:120]}  <{r['href']}>"
            )
        display = "\n".join(display_lines)

        # --- Page fetch ONLY for medium+ ---
        pages: dict[str, str] = {}
        if profile["pages"] > 0:
            page_urls = [r["href"] for _, r in top[:profile["pages"]]]
            pages = fetch_pages_parallel(page_urls, max_chars=profile["page_chars"])

        # --- Build context ---
        ctx_parts = []
        for rank, (_, r) in enumerate(top, 1):
            href  = r["href"]
            trust = domain_trust(href)
            # Truncate snippet for simple to keep prompt tiny
            snip  = r.get("body", "")
            if profile["pages"] == 0:
                snip = snip[:200]
            block = f"[Source {rank}, trust={trust}/5] <{href}>\n{snip}"
            if href in pages:
                block += f"\nPage: {pages[href]}"
            ctx_parts.append(block)

        context = "## Sources\n" + "\n\n".join(ctx_parts)
        return display, context

    except Exception as e:
        return f"Search failed: {e}", ""


# ---------------------------------------------------------------------------
# Tiered prompts — simple is minimal, complex is full
# ---------------------------------------------------------------------------

_SYS_SIMPLE = (
    "SYSTEM: Precise assistant. Answer from the sources below. "
    "Be concise. Do not invent unsourced facts."
)

_SYS_MEDIUM = (
    "SYSTEM: Precise assistant. Sources below are your primary truth.\n"
    "Rules: prefer high-trust sources; do not combine commands from "
    "different tools; if sources conflict, say so; if unsure say so."
)

_SYS_COMPLEX = (
    "SYSTEM: Precise assistant. Sources below are your PRIMARY and ONLY truth.\n"
    "## Grounding Rules (MANDATORY)\n"
    "- Base EVERY claim on provided sources. Cite which source.\n"
    "- Do NOT combine commands/syntax from different executables.\n"
    "- If sources conflict, state the conflict.\n"
    "- If not clearly in sources, say \"not confirmed by sources\".\n"
    "- Prefer trust=4-5 over trust=0-1.\n"
    "- Do NOT invent arguments, paths, or keys not in sources."
)

_SYS_NOSEARCH = (
    "SYSTEM: Precise assistant. Answer from training knowledge. "
    "If uncertain, say so."
)

_INSTR = {
    "code":    "Return working code in a fenced code block. No explanation unless asked.",
    "howto":   ("Concise step-by-step. Fenced code blocks for commands. "
                "1 sentence per step max. Common case first, then alternatives."),
    "fact":    "One sentence. One line of context max.",
    "general": "2-5 sentences max. Direct. No filler.",
}


def build_prompt(query: str, context: str, history: list[str],
                 complexity: str) -> str:
    qtype = classify_query(query)
    parts: list[str] = []

    if context:
        sys_prompt = {"simple": _SYS_SIMPLE, "medium": _SYS_MEDIUM,
                      "complex": _SYS_COMPLEX}[complexity]
        parts.append(f"{sys_prompt}\n\n{context}")
    else:
        parts.append(_SYS_NOSEARCH)

    if history:
        parts.append("## Conversation\n" + "\n".join(history))

    parts.append(f"## Q\n{query}\n## Instructions\n{_INSTR[qtype]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama — buffered streaming + keep_alive
# ---------------------------------------------------------------------------

def ask_ollama_stream(prompt: str, model: str, think_enabled: bool,
                      profile: dict) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature":    0.05,
            "num_predict":    profile["num_predict"],
            "num_ctx":        profile["num_ctx"],
            "top_k":          10,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
        },
    }
    if not think_enabled:
        payload["think"] = False

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **HTTP_HEADERS},
        method="POST",
    )
    chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            # Buffered line reading — massively faster than read(1)
            for raw_line in r:
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                token = obj.get("response", "")
                if token:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    chunks.append(token)
                if obj.get("done"):
                    break
        return "".join(chunks).strip()
    except Exception as e:
        msg = f"Ollama error: {e}"
        print(msg)
        return msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def print_interface(model: str, search: bool, think: bool) -> None:
    print("\n" + "=" * 56)
    print(f"  Model : {model}")
    print(f"  Search: {'ON  (trust-ranked, adaptive)' if search else 'OFF'}")
    print(f"  Think : {'VISIBLE' if think else 'HIDDEN'}")
    print("=" * 56)
    print("  /search  Toggle web search")
    print("  /think   Toggle model thinking visibility")
    print("  /model   Switch model  (e.g. /model llama3)")
    print("  /models  List local models")
    print("  /clear   Reset conversation history")
    print("  /help    Show this menu")
    print("  /exit    Quit")
    print("=" * 56 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",  default=DEFAULT_MODEL)
    parser.add_argument("-s", "--search", action="store_true")
    args = parser.parse_args()

    model: str           = args.model
    search_enabled: bool = args.search
    think_enabled: bool  = False
    history: list[str]   = []

    print_interface(model, search_enabled, think_enabled)

    # Pre-warm: load model into GPU/RAM now, not on first query
    print("  🔥 Warming up model...", file=sys.stderr)
    prewarm_model(model)

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
            elif cmd in ("/clear", "/forget"):
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
            elif cmd in ("/model", "/models"):
                local_models = fetch_local_models()
                if not local_models:
                    print("Could not reach Ollama.\n")
                    continue
                if len(parts) > 1:
                    target   = parts[1]
                    resolved = target if target in local_models else f"{target}:latest"
                    if resolved in local_models:
                        model = resolved
                        print("  🔥 Warming up model...", file=sys.stderr)
                        prewarm_model(model)
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

        # --- Adaptive complexity ---
        complexity = query_complexity(user_input)
        profile    = COMPLEXITY_PROFILE[complexity]

        # --- Resolve, search, respond ---
        search_query = resolve_query(user_input, history)
        context_str  = ""

        if search_enabled:
            label = "snippets" if profile["pages"] == 0 else f"{profile['pages']} pages"
            print(f"🔍 Searching ({complexity}, {label})...", file=sys.stderr)
            t_search = time.time()
            display, context_str = web_search(search_query, profile)
            t_search = time.time() - t_search
            if context_str:
                print(f"\n🌐 [Search — {label} in {t_search:.1f}s]")
                print(display)
                print("-" * 56)

        full_prompt = build_prompt(user_input, context_str, history, complexity)

        print()
        t0       = time.time()
        response = ask_ollama_stream(full_prompt, model, think_enabled, profile)
        elapsed  = time.time() - t0
        print(f"\n⏱️  {elapsed:.1f}s  [{complexity}]\n")

        if "Ollama error" not in response:
            history.append(f"User: {user_input}")
            history.append(f"AI: {response}")
            if len(history) > 6:
                history = history[-6:]


if __name__ == "__main__":
    main()