#!/usr/bin/env python3
"""
ask.py - Fast interactive Ollama CLI with web search, status, think control,
multiline paste, and cancel breaker.
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
import urllib.request
from typing import Any

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:latest"
MODEL_KEEP_ALIVE = "-1m"
OLLAMA_TIMEOUT = 300
CONNECT_TIMEOUT = 3
SEARCH_CACHE_TTL_SECONDS = 600

DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"

CANCEL_EVENT = threading.Event()
SEARCH_CACHE: dict[str, tuple[float, str, str]] = {}
SESSION_HISTORY: list[dict[str, str]] = []

SYSTEM_PROMPT = (
    "You are a precise, fast local assistant. Analyze the user prompt for any specific context keys "
    "like geographical location, operating system, application version, or calendar date. If the provided "
    "search snippets contain conflicting context keys (such as a different province, version number, or year), "
    "immediately disregard those specific numbers or instructions. Prioritize internal training knowledge "
    "when search context does not perfectly match the specific boundaries defined in the user's prompt."
)

class Cancelled(Exception):
    pass

class Breaker:
    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event, self.stop_event = cancel_event, threading.Event()
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

    def _watch(self) -> None:
        if self.is_windows:
            try:
                import msvcrt
                while not self.stop_event.is_set() and not self.cancel_event.is_set():
                    if msvcrt.kbhit() and msvcrt.getch() in (b"\x1b", b"\x03", b"\x04"):
                        self.cancel_event.set()
                    time.sleep(0.03)
            except Exception: pass
        else:
            try:
                import select, termios, tty
                fd = sys.stdin.fileno()
                if os.isatty(fd):
                    self.old_settings = termios.tcgetattr(fd)
                    tty.setcbreak(fd)
                    while not self.stop_event.is_set() and not self.cancel_event.is_set():
                        if select.select([sys.stdin], [], [], 0.05)[0] and sys.stdin.read(1) in ("\x1b", "\x03", "\x04"):
                            self.cancel_event.set()
            except Exception: pass
            finally: self._restore_terminal()

    def _restore_terminal(self) -> None:
        if not self.is_windows and self.old_settings:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)
                self.old_settings = None
            except Exception: pass

class Spinner:
    def __init__(self, text: str = "Generating") -> None:
        self.text = text
        self.stop_event = threading.Event()

    def start(self) -> "Spinner":
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        print("\r" + " " * 80 + "\r", end="", flush=True)

    def _run(self) -> None:
        frames, i = ["|", "/", "-", "\\"], 0
        while not self.stop_event.is_set() and not CANCEL_EVENT.is_set():
            print(f"\r{frames[i % 4]} {self.text}... Esc/Ctrl+C/Ctrl+D to cancel", end="", flush=True)
            i += 1
            time.sleep(0.08)

def handle_sigint(signum, frame) -> None:
    CANCEL_EVENT.set()
    raise KeyboardInterrupt

def cancel_if_requested() -> None:
    if CANCEL_EVENT.is_set():
        raise Cancelled

def ollama_url(path: str) -> str:
    return f"{OLLAMA_BASE.rstrip('/')}/{path.lstrip('/')}"

def is_fresh(timestamp: float, ttl: int) -> bool:
    return time.time() - timestamp <= ttl

def fetch_local_models() -> list[str]:
    try:
        req = urllib.request.Request(ollama_url("/api/tags"), headers={"User-Agent": "ask-cli"})
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as response:
            data = json.loads(response.read())
        return sorted(model["name"] for model in data.get("models", []) if model.get("name"))
    except Exception:
        return []

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

def web_search(query: str) -> tuple[str, str]:
    cache_key = query.strip().lower()
    cached = SEARCH_CACHE.get(cache_key)
    if cached and is_fresh(cached[0], SEARCH_CACHE_TTL_SECONDS):
        return cached[1], cached[2]
    cancel_if_requested()
    try:
        from ddgs import DDGS
    except Exception:
        return "", ""
    try:
        with DDGS(timeout=1.5) as ddgs:
            raw = ddgs.text(query, max_results=5)
    except Exception:
        return "", ""
    cancel_if_requested()
    if not raw:
        return "", ""
    
    # Sentence-Bound Chunking
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    sentences = []
    
    for idx, item in enumerate(raw, start=1):
        cancel_if_requested()
        url = item.get("href") or item.get("url") or ""
        title = item.get("title") or "Untitled"
        body = item.get("body", "").strip()
        if not body or not url:
            continue
        parts = re.split(r'(?<=[.!?])\s+', body)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append({
                    "text": p,
                    "url": url.strip(),
                    "title": title.strip(),
                    "idx": idx
                })
                
    # Programmatic Token-Overlap Scorer
    scored_sentences = []
    for sent_info in sentences:
        sent_tokens = set(re.findall(r'\b\w+\b', sent_info["text"].lower()))
        score = len(query_tokens.intersection(sent_tokens))
        scored_sentences.append((score, sent_info))
        
    # Precision Ingestion (top 4 sentences)
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = scored_sentences[:4]
    
    if not top_sentences:
        return "", ""
        
    context_parts = []
    sources = []
    seen_urls = set()
    for rank, (score, sent_info) in enumerate(top_sentences, start=1):
        context_parts.append(
            f"[{rank}] Source: {sent_info['title']} | {sent_info['url']}\n"
            f"Snippet: {sent_info['text']}"
        )
        if sent_info['url'] not in seen_urls:
            seen_urls.add(sent_info['url'])
            sources.append(f"[{rank}] {sent_info['title']} | {sent_info['url']}")
            
    context = "\n\n".join(context_parts)
    source_text = "\n".join(sources)
    SEARCH_CACHE[cache_key] = (time.time(), context, source_text)
    return context, source_text

def ask_ollama_stream(messages: list[dict[str, str]], model: str, think_enabled: bool) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": MODEL_KEEP_ALIVE,
        "think": think_enabled,
        "options": {
            "temperature": 0.05,
            "num_ctx": 3072,
            "top_k": 10,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        },
    }
    req = urllib.request.Request(
        ollama_url("/api/chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ask-cli"},
        method="POST"
    )
    response_chunks: list[str] = []
    hidden_thinking_seen = False
    thinking_active = False
    first_token = True
    spinner = None
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
                
                message = obj.get("message", {})
                think_token = message.get("thinking", "")
                response_token = message.get("content", "")
                
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
        # pyrefly: ignore [missing-import]
        from prompt_toolkit import PromptSession
        # pyrefly: ignore [missing-import]
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
    print(f" Search: {'ON' if search else 'OFF'}")
    print(f" Think : {'VISIBLE' if think else 'HIDDEN'}")
    print("=" * 56)
    print(" /search Toggle web search")
    print(" /think  Toggle thinking mode and visibility")
    print(" /model  List and switch models")
    print(" /model <name> Switch model directly")
    print(" /status Show keep_alive, search, think, cache, and history status")
    print(" /clear  Reset conversation history")
    print(" /help   Show this menu")
    print(" /exit   Quit")
    print(" Paste multiline text directly. Enter sends. Ctrl+J or Esc+Enter adds a new line.")
    print(" Esc, Ctrl+C, or Ctrl+D cancels active processing")
    print("=" * 56 + "\n")

def print_status(model: str, search: bool, think: bool) -> None:
    print("\nStatus")
    print(f" Model     : {model}")
    print(f" Keep alive: {MODEL_KEEP_ALIVE}")
    print(f" Search    : {'ON' if search else 'OFF'}")
    print(f" Think     : {'VISIBLE' if think else 'HIDDEN'}")
    print(f" Cache     : search={len(SEARCH_CACHE)}")
    print(f" History   : {len(SESSION_HISTORY)} messages ({len(SESSION_HISTORY)//2} turns)")

def main() -> None:
    signal.signal(signal.SIGINT, handle_sigint)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-s", "--search", action="store_true")
    args = parser.parse_args()
    model = args.model
    search_enabled = args.search
    think_enabled = False
    print_interface(model, search_enabled, think_enabled)
    
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
            SESSION_HISTORY.clear()
            print("Conversation history cleared.")
            continue
        if command_lower == "/status":
            print_status(model, search_enabled, think_enabled)
            continue
        if command_lower == "/model":
            model = arg if arg else switch_model_interactive(model)
            print(f"Model: {model}")
            continue
            
        try:
            with Breaker(CANCEL_EVENT):
                context = ""
                sources = ""
                if search_enabled:
                    print(f"Searching... Esc/Ctrl+C/Ctrl+D to cancel", file=sys.stderr)
                    context, sources = web_search(user_input)
                    cancel_if_requested()
                    if sources:
                        print("Sources:\n" + sources, file=sys.stderr)
                
                # Prepare message array for /api/chat
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(SESSION_HISTORY)
                
                if context:
                    user_content = f"Sources:\n{context}\n\nUser question: {user_input}"
                else:
                    user_content = user_input
                    
                messages.append({"role": "user", "content": user_content})
                
                started = time.time()
                answer = ask_ollama_stream(messages, model, think_enabled)
                elapsed = time.time() - started
                cancel_if_requested()
                
            if answer:
                print(f"\nTime: {elapsed:.1f}s\n")
                if not answer.startswith("Ollama error") and not answer.startswith("No final response") and not answer == "Cancelled.":
                    SESSION_HISTORY.append({"role": "user", "content": user_content})
                    SESSION_HISTORY.append({"role": "assistant", "content": answer})
                    while len(SESSION_HISTORY) > 6:
                        SESSION_HISTORY.pop(0)
        except (Cancelled, KeyboardInterrupt, EOFError):
            CANCEL_EVENT.set()
            print("\nCancelled. Start over when ready.")
            continue
        finally:
            CANCEL_EVENT.clear()

if __name__ == "__main__":
    main()
