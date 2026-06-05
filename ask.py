#!/usr/bin/env python3
"""
ask.py - Fast interactive Ollama CLI with web search, quiet background warmup,
status, think control, multiline paste, and cancel breaker.

Usage:
  python ask.py [-m model] [-s]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from html.parser import HTMLParser
from typing import Any

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "phi4-mini:latest"
MODEL_KEEP_ALIVE= "-1m"
OLLAMA_TIMEOUT = 300
CONNECT_TIMEOUT = 3
PAGE_TIMEOUT = 3
SEARCH_PAGE_TIMEOUT = 5
PAGE_BYTE_LIMIT = 12_000
PAGE_CACHE_TTL_SECONDS = 900
SEARCH_CACHE_TTL_SECONDS = 600
THINK_BUDGET_MULTIPLIER = 3
MAX_HISTORY_ITEMS = 12

DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"

CANCEL_EVENT = threading.Event()
WARMUP_LOCK = threading.Lock()
WARMING_MODELS: set[str] = set()
WARMED_MODELS: set[str] = set()
WARMUP_FAILED: dict[str, str] = {}
PAGE_CACHE: dict[str, tuple[float, str]] = {}
SEARCH_CACHE: dict[tuple[str, str], tuple[float, str, str]] = {}

BLOCKED_DOMAINS = {
    "wikipedia.org", "wikimedia.org", "wikidata.org", "wikihow.com",
    "wikia.com", "fandom.com", "wiki.org", "quora.com", "reddit.com",
    "answers.com", "ask.com", "britannica.com", "encyclopedia.com",
    "infoplease.com", "about.com", "reference.com", "thoughtco.com",
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

COMPLEXITY_PROFILE = {
    "simple": dict(num_predict=192, num_ctx=1536, snippets=3, search_results=5, pages=0, page_chars=0),
    "medium": dict(num_predict=384, num_ctx=3072, snippets=5, search_results=8, pages=1, page_chars=2200),
    "complex": dict(num_predict=900, num_ctx=4096, snippets=8, search_results=12, pages=3, page_chars=3000),
}

EXPLICIT_CODE = {"python", "bash", "shell", "script", "code", "function", "snippet", "oneliner", "one-liner", "regex", "sql", "js", "javascript", "typescript", "rust", "go", "c++", "java", "powershell", "cmdlet"}
HOWTO_TOKENS = {"how to", "how do i", "how can i", "steps to", "way to", "command"}
FACT_TOKENS = {"what is", "who is", "when did", "where is", "define", "meaning of", "version", "latest", "current", "price", "release date"}
COMPARISON_TOKENS = {"difference between", "differences between", "compare", "comparison", "versus", " vs ", "pros and cons", "better", "worse", "advantages", "disadvantages"}
REFERENTIAL_STARTS = {"that", "it", "this", "those", "them", "there", "the same", "more", "more about", "what about", "and also", "also", "latest on", "why", "elaborate", "explain"}

_SYS_SIMPLE = "SYSTEM: Precise assistant. Answer from the sources below. Be concise. Do not invent unsourced facts."
_SYS_MEDIUM = "SYSTEM: Precise assistant. Sources below are your primary truth. Prefer high-trust sources. If unsure, say so."
_SYS_COMPLEX = "SYSTEM: Precise assistant. Sources below are your PRIMARY and ONLY truth. Cite source numbers. Prefer trust=4-5. If not clearly in sources, say not confirmed by sources."
_SYS_NOSEARCH = "SYSTEM: Precise assistant. Answer from training knowledge. If uncertain, say so."

_INSTR = {
    "code": "Return working code in a fenced code block. No explanation unless asked.",
    "howto": "Concise step-by-step. Fenced code blocks for commands. One sentence per step max.",
    "fact": "One sentence. One line of context max.",
    "general": "2-5 sentences max. Direct. No filler.",
}
_INSTR_THINK = {
    "code": "Return working code in a fenced code block. Brief explanation is okay.",
    "howto": "Clear step-by-step. Fenced code blocks for commands. Common case first, then alternatives.",
    "fact": "State the fact clearly. Add brief context if helpful.",
    "general": "Answer thoroughly but concisely. Use structure if it helps clarity.",
}

class Cancelled(Exception):
    pass

class Breaker:
    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.is_windows = platform.system().lower().startswith("win")
        self.old_settings = None

    def __enter__(self) -> "Breaker":
        self.cancel_event.clear()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_event.set()
        self._restore_terminal()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.4)

    def _watch(self) -> None:
        self._watch_windows() if self.is_windows else self._watch_posix()

    def _watch_windows(self) -> None:
        try:
            import msvcrt
            while not self.stop_event.is_set() and not self.cancel_event.is_set():
                if msvcrt.kbhit() and msvcrt.getch() in (b"\x1b", b"\x03", b"\x04"):
                    self.cancel_event.set()
                    break
                time.sleep(0.03)
        except Exception:
            return

    def _watch_posix(self) -> None:
        try:
            import select
            import termios
            import tty
            fd = sys.stdin.fileno()
            if not os.isatty(fd):
                return
            self.old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while not self.stop_event.is_set() and not self.cancel_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready and sys.stdin.read(1) in ("\x1b", "\x03", "\x04"):
                    self.cancel_event.set()
                    break
        except Exception:
            return
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self.is_windows or self.old_settings is None:
            return
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)
            self.old_settings = None
        except Exception:
            pass

class Spinner:
    _FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, text: str = "Generating") -> None:
        self.text = text
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> "Spinner":
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.4)
        print("\r" + " " * 90 + "\r", end="", flush=True)

    def _run(self) -> None:
        i = 0
        while not self.stop_event.is_set() and not CANCEL_EVENT.is_set():
            print(f"\r{self._FRAMES[i % len(self._FRAMES)]} {self.text}... Esc/Ctrl+C/Ctrl+D to cancel", end="", flush=True)
            i += 1
            time.sleep(0.08)

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
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._chunks)

def handle_sigint(signum, frame) -> None:
    CANCEL_EVENT.set()
    raise KeyboardInterrupt

def cancel_if_requested() -> None:
    if CANCEL_EVENT.is_set():
        raise Cancelled

def ollama_url(path: str) -> str:
    return f"{OLLAMA_BASE.rstrip('/')}/{path.lstrip('/')}"

def keep_alive_value() -> str | int:
    if isinstance(MODEL_KEEP_ALIVE, int):
        return MODEL_KEEP_ALIVE
    value = str(MODEL_KEEP_ALIVE).strip()
    return -1 if value == "-1" else value

def is_fresh(timestamp: float, ttl: int) -> bool:
    return time.time() - timestamp <= ttl

def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", parser.get_text()).strip()

def is_blocked(url: str) -> bool:
    try:
        host = re.sub(r"^www\.", "", urllib.parse.urlparse(url).netloc.lower())
        return any(host == domain or host.endswith("." + domain) for domain in BLOCKED_DOMAINS)
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
        req = urllib.request.Request(ollama_url("/api/tags"), headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as response:
            data = json.loads(response.read())
        return sorted(model["name"] for model in data.get("models", []) if model.get("name"))
    except Exception:
        return []

def model_exists_locally(model: str) -> bool:
    models = fetch_local_models()
    return model in models or (model.endswith(":latest") and model.replace(":latest", "") in models)

def switch_model_interactive(current_model: str) -> str:
    models = fetch_local_models()
    if not models:
        print("No local models found or Ollama is not reachable.")
        return current_model
    print("\nLocal models:")
    for idx, name in enumerate(models, start=1):
        print(f" {idx:>2}. {name}{' *' if name == current_model else ''}")
    choice = input("\nPick model number or name, Enter to cancel: ").strip()
    if not choice:
        print("Model switch cancelled.")
        return current_model
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    if choice in models:
        return choice
    matches = [model for model in models if choice.lower() in model.lower()]
    if len(matches) == 1:
        return matches[0]
    print("Multiple matches. Use the full name or model number." if matches else "Model not found locally.")
    return current_model

def prewarm_model(model: str) -> tuple[bool, str]:
    payload = {
        "model": model,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1, "num_ctx": 32, "temperature": 0},
        "keep_alive": keep_alive_value(),
    }
    req = urllib.request.Request(
        ollama_url("/api/generate"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            response.read()
        return True, "complete"
    except Exception as exc:
        return False, str(exc)

def start_background_warmup(model: str) -> None:
    with WARMUP_LOCK:
        if model in WARMED_MODELS or model in WARMING_MODELS:
            return
        WARMING_MODELS.add(model)
        WARMUP_FAILED.pop(model, None)

    def worker() -> None:
        try:
            ok, message = prewarm_model(model)
            with WARMUP_LOCK:
                if ok:
                    WARMED_MODELS.add(model)
                    WARMUP_FAILED.pop(model, None)
                else:
                    WARMUP_FAILED[model] = message
        finally:
            with WARMUP_LOCK:
                WARMING_MODELS.discard(model)

    threading.Thread(target=worker, daemon=True).start()

def print_status(model: str, search: bool, think: bool) -> None:
    with WARMUP_LOCK:
        if model in WARMED_MODELS:
            warmup = "complete"
        elif model in WARMING_MODELS:
            warmup = "warming"
        elif model in WARMUP_FAILED:
            warmup = "failed"
        else:
            warmup = "not warmed"
        error = WARMUP_FAILED.get(model, "")
    print("\nStatus")
    print(f" Model     : {model}")
    print(f" Warmup    : {warmup}")
    if error:
        print(f" Warmup err: {error}")
    print(f" Keep alive: {keep_alive_value()}")
    print(f" Search    : {'ON' if search else 'OFF'}")
    print(f" Think     : {'VISIBLE' if think else 'HIDDEN'}")
    print(f" Cache     : search={len(SEARCH_CACHE)}, pages={len(PAGE_CACHE)}")

def fetch_page(url: str, byte_limit: int = PAGE_BYTE_LIMIT) -> str:
    cached = PAGE_CACHE.get(url)
    if cached and is_fresh(cached[0], PAGE_CACHE_TTL_SECONDS):
        return cached[1]
    try:
        cancel_if_requested()
        if is_blocked(url):
            return ""
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=PAGE_TIMEOUT) as response:
            cancel_if_requested()
            raw = response.read(byte_limit).decode("utf-8", errors="ignore")
        text = html_to_text(raw)
        if text:
            PAGE_CACHE[url] = (time.time(), text)
        return text
    except Cancelled:
        raise
    except Exception:
        return ""

def fetch_pages_parallel(urls: list[str], max_chars: int = 2500, workers: int = 4) -> dict[str, str]:
    results: dict[str, str] = {}
    if not urls or CANCEL_EVENT.is_set():
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_page, url): url for url in urls}
        try:
            for future in as_completed(futures, timeout=SEARCH_PAGE_TIMEOUT):
                cancel_if_requested()
                url = futures[future]
                try:
                    text = future.result()
                    if text:
                        results[url] = text[:max_chars]
                except Cancelled:
                    raise
                except Exception:
                    pass
        except TimeoutError:
            pass
    return results

def _content_words(text: str) -> list[str]:
    return [word for word in re.sub(r"[^\w\s]", "", text.lower()).split() if word not in STOP_WORDS]

def query_complexity(query: str) -> str:
    query_lower = query.lower()
    words = _content_words(query)
    if len(words) <= 5 and not any(token in query_lower for token in COMPARISON_TOKENS):
        return "simple"
    return "medium" if len(words) <= 14 else "complex"

def classify_query(query: str) -> str:
    query_lower = query.lower()
    words = set(query_lower.split())
    if any(token in query_lower for token in COMPARISON_TOKENS):
        return "general"
    if words & EXPLICIT_CODE and not any(token in query_lower for token in HOWTO_TOKENS):
        return "code"
    if any(token in query_lower for token in HOWTO_TOKENS):
        return "howto"
    if any(token in query_lower for token in FACT_TOKENS):
        return "fact"
    return "general"

def resolve_query(query: str, history: list[str]) -> str:
    if not history:
        return query
    query_lower = query.lower().strip()
    explicitly_referential = any(query_lower.startswith(ref) for ref in REFERENTIAL_STARTS)
    pronoun_only = bool(re.match(r"^(it|that|this|those|them)\b", query_lower))
    if len(_content_words(query)) >= 2 and not explicitly_referential and not pronoun_only:
        return query
    last_user = next((entry[5:].strip() for entry in reversed(history) if entry.startswith("User:")), "")
    if not last_user:
        return query
    topic = " ".join(_content_words(last_user)[:8])
    if not topic:
        return query
    resolved = f"Regarding {topic}: {query}"
    print(f"Resolved: {resolved}", file=sys.stderr)
    return resolved

def _keyword_score(query_words: set[str], title: str, body: str) -> float:
    title_lower = title.lower()
    body_lower = body.lower()
    return sum(2 for word in query_words if word in title_lower) + sum(1 for word in query_words if word in body_lower)

def web_search(query: str, profile: dict) -> tuple[str, str]:
    cache_key = (query.strip().lower(), json.dumps(profile, sort_keys=True))
    cached = SEARCH_CACHE.get(cache_key)
    if cached and is_fresh(cached[0], SEARCH_CACHE_TTL_SECONDS):
        return cached[1], cached[2]
    cancel_if_requested()
    try:
        from ddgs import DDGS
    except Exception:
        return "Web search unavailable. Install with: pip install ddgs", ""
    try:
        raw = DDGS().text(query, max_results=profile["search_results"])
    except Exception as exc:
        return f"Search failed: {exc}", ""
    cancel_if_requested()
    if not raw:
        return "No results found.", ""
    query_words = set(_content_words(query))
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        cancel_if_requested()
        url = item.get("href") or item.get("url") or ""
        title = item.get("title") or "Untitled"
        body = item.get("body") or ""
        if not url or is_blocked(url):
            continue
        cleaned.append({"title": title.strip(), "body": body.strip(), "url": url.strip(), "trust": domain_trust(url), "score": _keyword_score(query_words, title, body)})
    cleaned.sort(key=lambda item: (item["trust"], item["score"]), reverse=True)
    cleaned = cleaned[:profile["snippets"]]
    if not cleaned:
        return "No usable results found.", ""
    page_urls = [item["url"] for item in cleaned[:profile["pages"]]]
    page_text = fetch_pages_parallel(page_urls, max_chars=profile["page_chars"]) if page_urls else {}
    context_parts = []
    sources = []
    for idx, item in enumerate(cleaned, start=1):
        cancel_if_requested()
        source_line = f"[{idx}] {item['title']} | trust={item['trust']} | {item['url']}"
        context_parts.append(f"{source_line}\nSnippet: {item['body']}\nPage: {page_text.get(item['url'], '')}".strip())
        sources.append(source_line)
    context = "\n\n".join(context_parts)
    source_text = "\n".join(sources)
    SEARCH_CACHE[cache_key] = (time.time(), context, source_text)
    return context, source_text

def build_prompt(query: str, context: str, history: list[str], complexity: str, think_enabled: bool) -> str:
    query_type = classify_query(query)
    system_prompt = {"simple": _SYS_SIMPLE, "medium": _SYS_MEDIUM, "complex": _SYS_COMPLEX}[complexity] if context else _SYS_NOSEARCH
    instruction = _INSTR_THINK[query_type] if think_enabled else _INSTR[query_type]
    history_text = "\n".join(history[-MAX_HISTORY_ITEMS:])
    parts = [system_prompt, f"Instruction: {instruction}"]
    if history_text:
        parts.append(f"Conversation history:\n{history_text}")
    if context:
        parts.append(f"Sources:\n{context}")
    parts.append(f"User question:\n{query}")
    parts.append("Assistant answer:")
    return "\n\n".join(parts)

def ask_ollama_stream(prompt: str, model: str, think_enabled: bool, profile: dict) -> str:
    num_predict = profile["num_predict"] * (THINK_BUDGET_MULTIPLIER if think_enabled else 1)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive_value(),
        "think": think_enabled,
        "options": {"temperature": 0.05, "num_predict": num_predict, "num_ctx": profile["num_ctx"], "top_k": 10, "top_p": 0.9, "repeat_penalty": 1.1},
    }
    req = urllib.request.Request(ollama_url("/api/generate"), data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", **HTTP_HEADERS}, method="POST")
    response_chunks: list[str] = []
    hidden_thinking_seen = False
    thinking_active = False
    first_token = True
    spinner: Spinner | None = None
    if not think_enabled:
        spinner = Spinner("Generating").start()
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            for raw_line in response:
                cancel_if_requested()
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                think_token = obj.get("thinking", "")
                if think_token:
                    if think_enabled:
                        if first_token:
                            first_token = False
                            if spinner:
                                spinner.stop()
                                spinner = None
                        if not thinking_active:
                            thinking_active = True
                            sys.stdout.write(f"\n{DIM}{CYAN}Thinking...{RESET}\n{DIM}")
                            sys.stdout.flush()
                        sys.stdout.write(think_token)
                        sys.stdout.flush()
                    else:
                        hidden_thinking_seen = True
                response_token = obj.get("response", "")
                if response_token:
                    if first_token:
                        first_token = False
                        if spinner:
                            spinner.stop()
                            spinner = None
                    if thinking_active:
                        thinking_active = False
                        sys.stdout.write(f"{RESET}\n\n")
                        sys.stdout.flush()
                    sys.stdout.write(response_token)
                    sys.stdout.flush()
                    response_chunks.append(response_token)
                if obj.get("done"):
                    if spinner:
                        spinner.stop()
                        spinner = None
                    if thinking_active:
                        sys.stdout.write(f"{RESET}\n\n")
                        sys.stdout.flush()
                    break
        if spinner:
            spinner.stop()
        response_text = "".join(response_chunks).strip()
        if response_text:
            print()
            return response_text
        msg = "No final response returned. The model sent thinking output only, but /think is OFF. Try /think ON, or use a non-reasoning model." if hidden_thinking_seen and not think_enabled else "No final response returned by the model. Try again or switch models."
        print(msg)
        return msg
    except Cancelled:
        if spinner:
            spinner.stop()
        print("\nCancelled.")
        return ""
    except KeyboardInterrupt:
        CANCEL_EVENT.set()
        if spinner:
            spinner.stop()
        print("\nCancelled.")
        return ""
    except Exception as exc:
        if spinner:
            spinner.stop()
        msg = f"Ollama error: {exc}"
        print(msg)
        return msg

def read_user_input() -> str | None:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        @kb.add("c-j")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        session = PromptSession(multiline=True, key_bindings=kb, prompt_continuation="... ")
        return session.prompt("\nYou> ").strip()
    except ImportError:
        try:
            return input("\nYou> ").strip()
        except EOFError:
            return None
        except KeyboardInterrupt:
            print("\nUse /exit to quit.")
            return ""
    except EOFError:
        return None
    except KeyboardInterrupt:
        print("\nUse /exit to quit.")
        return ""

def print_interface(model: str, search: bool, think: bool) -> None:
    print("\n" + "=" * 56)
    print(f" Model : {model}")
    print(f" Search: {'ON  (trust-ranked, adaptive)' if search else 'OFF'}")
    print(f" Think : {'VISIBLE' if think else 'HIDDEN'}")
    print("=" * 56)
    print(" /search Toggle web search")
    print(" /think Toggle thinking mode and visibility")
    print(" /model List and switch models")
    print(" /model <name> Switch model directly")
    print(" /warm Warm current model quietly in background")
    print(" /status Show warmup, keep_alive, search, think, and cache status")
    print(" /clear Reset conversation history")
    print(" /help Show this menu")
    print(" /exit Quit")
    print(" Paste multiline text directly. Enter sends. Ctrl+J or Esc+Enter adds a new line.")
    print(" Background warmup is quiet. Use /status to check it.")
    print(" Esc, Ctrl+C, or Ctrl+D cancels active processing")
    print("=" * 56 + "\n")

def main() -> None:
    signal.signal(signal.SIGINT, handle_sigint)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-s", "--search", action="store_true")
    args = parser.parse_args()
    model = args.model
    search_enabled = args.search
    think_enabled = False
    history: list[str] = []
    print_interface(model, search_enabled, think_enabled)
    start_background_warmup(model)
    while True:
        user_input = read_user_input()
        if user_input is None:
            print("\nBye.")
            break
        if not user_input:
            continue
        command, _, arg = user_input.partition(" ")
        command_lower = command.lower()
        arg = arg.strip()
        if command_lower in {"/exit", "/quit", "exit", "quit"}:
            print("Bye.")
            break
        if command_lower == "/help":
            print_interface(model, search_enabled, think_enabled)
            continue
        if command_lower == "/search":
            search_enabled = not search_enabled
            print(f"Search {'ON' if search_enabled else 'OFF'}")
            continue
        if command_lower == "/think":
            think_enabled = not think_enabled
            print(f"Think {'ON and visible' if think_enabled else 'OFF and hidden'}")
            continue
        if command_lower == "/clear":
            history.clear()
            print("Conversation history cleared.")
            continue
        if command_lower == "/warm":
            start_background_warmup(model)
            print("Warmup started quietly. Use /status to check progress.")
            continue
        if command_lower == "/status":
            print_status(model, search_enabled, think_enabled)
            continue
        if command_lower == "/models":
            print("/models has been merged into /model. Use /model to list and switch models.")
            model = switch_model_interactive(model)
            print(f"Model: {model}")
            start_background_warmup(model)
            continue
        if command_lower == "/model":
            model = arg if arg else switch_model_interactive(model)
            print(f"Model: {model}")
            start_background_warmup(model)
            continue
        resolved_query = resolve_query(user_input, history)
        complexity = query_complexity(resolved_query)
        profile = COMPLEXITY_PROFILE[complexity]
        try:
            with Breaker(CANCEL_EVENT):
                context = ""
                sources = ""
                if search_enabled:
                    label = "snippets" if profile["pages"] == 0 else f"{profile['pages']} pages"
                    print(f"Searching ({complexity}, {label})... Esc/Ctrl+C/Ctrl+D to cancel", file=sys.stderr)
                    context, sources = web_search(resolved_query, profile)
                    cancel_if_requested()
                    if sources:
                        print("Sources:\n" + sources, file=sys.stderr)
                prompt = build_prompt(resolved_query, context, history, complexity, think_enabled)
                started = time.time()
                answer = ask_ollama_stream(prompt, model, think_enabled, profile)
                elapsed = time.time() - started
                cancel_if_requested()
            if answer:
                budget = profile["num_predict"] * (THINK_BUDGET_MULTIPLIER if think_enabled else 1)
                print(f"\nTime: {elapsed:.1f}s [{complexity}, budget={budget}]\n")
                if not answer.startswith("Ollama error") and not answer.startswith("No final response"):
                    history.append(f"User: {user_input}")
                    history.append(f"Assistant: {answer}")
                    history = history[-MAX_HISTORY_ITEMS:]
        except (Cancelled, KeyboardInterrupt, EOFError):
            CANCEL_EVENT.set()
            print("\nCancelled. Start over when ready.")
            continue
        finally:
            CANCEL_EVENT.clear()

if __name__ == "__main__":
    main()
