#!/usr/bin/env python3
"""
ask.py - Minimal Interactive Ollama CLI with response benchmarking
Usage: python ask.py [-m model] [-s]
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:latest"

def fetch_local_models() -> list:
    try:
        url = f"{OLLAMA_BASE}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []

def web_search(query: str) -> str:
    url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + "&format=json&no_html=1"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        answer = data.get("AbstractText") or data.get("Answer") or ""
        related = [r["Text"] for r in data.get("RelatedTopics", []) if "Text" in r][:3]
        return (answer + "\n" + "\n".join(related)).strip() or "No instant answer found."
    except Exception as e:
        return f"Search failed: {e}"

def ask_ollama(prompt: str, model: str, think_enabled: bool) -> str:
    payload_dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256, "num_ctx": 2048, "top_k": 10, "top_p": 0.9}
    }
    if not think_enabled:
        payload_dict["think"] = False

    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        return f"Ollama error: {e}"

def print_interface(model: str, search: bool, think: bool):
    print("\n" + "="*50)
    print(f"Model: {model} | Search: {'ON' if search else 'OFF'} | Thinking: {'VISIBLE' if think else 'HIDDEN'}")
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

    model = args.model
    search_enabled = args.search
    think_enabled = False
    history = []

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
            cmd = parts[0].lower()
            
            if cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd in ["/clear", "/forget"]:
                history = []
                print("🧠 Conversation history cleared.\n")
                continue
            elif cmd == "/help":
                print_interface(model, search_enabled, think_enabled)
                continue
            elif cmd == "/search":
                search_enabled = not search_enabled
                print_interface(model, search_enabled, think_enabled)
                continue
            elif cmd == "/think":
                think_enabled = not think_enabled
                print_interface(model, search_enabled, think_enabled)
                continue
            elif cmd in ["/model", "/models"]:
                local_models = fetch_local_models()
                if not local_models:
                    print("⚠️ Could not retrieve local models.\n")
                    continue
                
                if len(parts) > 1:
                    target_model = parts[1]
                    if target_model in local_models or f"{target_model}:latest" in local_models:
                        model = target_model if target_model in local_models else f"{target_model}:latest"
                        print_interface(model, search_enabled, think_enabled)
                    else:
                        print(f"❌ Model '{target_model}' not found locally.\n")
                else:
                    print("\nLocal Models:")
                    for m in local_models:
                        print(f"  {'*' if m == model else ' '} {m}")
                    print("\n💡 Type '/model [name]' to switch.\n")
                continue
            else:
                print(f"Unknown command: {cmd}\n")
                continue

        # Build Context
        context_str = ""
        if search_enabled:
            print("🔍 Searching...", file=sys.stderr)
            search_context = web_search(user_input)
            if search_context and "Search failed" not in search_context:
                context_str += f"Web Context:\n{search_context}\n\n"

        if history:
            context_str += "Previous Conversation:\n" + "\n".join(history) + "\n\n"

        full_prompt = f"{context_str}Question: {user_input}\n\nGive a short, accurate answer."

        # Fetch Response and benchmark performance
        print(f"🤖 [{model}] Thinking...", file=sys.stderr, end="\r")
        
        start_time = time.time()
        response = ask_ollama(full_prompt, model, think_enabled)
        elapsed_time = time.time() - start_time
        
        # Clear the "Thinking..." line
        print(" " * 50, end="\r", file=sys.stderr)
        
        # Output result with execution duration
        print(f"\n{response}")
        print(f"⏱️ Time taken: {elapsed_time:.2f}s\n")

        if "Ollama error" not in response:
            history.append(f"User: {user_input}")
            history.append(f"AI: {response}")
            if len(history) > 6:
                history = history[-6:]

if __name__ == "__main__":
    main()
