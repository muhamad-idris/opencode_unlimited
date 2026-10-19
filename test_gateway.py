import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def post(path, data, stream=False):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        return e.code, None

    return status, raw


def test_models():
    print("=== GET /v1/models ===")
    req = urllib.request.Request(f"{BASE_URL}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            print(f"Total models: {len(models)}")
            print(f"First 10: {models[:10]}")
            print(f"Has nemotron-3.5-super-vl-free: "
                  f"{'nemotron-3.5-super-vl-free' in models}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_chat_nonstream():
    print("=== POST /v1/chat/completions (non-stream) ===")
    payload = {
        "model": "nemotron-3.5-super-vl-free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Halo, apa kabar? Balas dalam Bahasa Indonesia."},
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    payload = json.load(open('opencode_request_body.json','r'))
    status, raw = post("/v1/chat/completions", payload)
    if raw:
        try:
            data = json.loads(raw.decode("utf-8"))
            choice = data.get("choices", [{}])[0] if data.get("choices") else {}
            msg = choice.get("message", {})
            print(f"Status: {status}")
            print(f"Model: {data.get('model')}")
            print(f"Finish reason: {choice.get('finish_reason')}")
            print(f"Content: {msg.get('content', '')[:500]}")
            usage = data.get("usage")
            if usage:
                print(f"Usage: {usage}")
        except json.JSONDecodeError:
            print(f"Status: {status}")
            print(f"Raw (first 500 chars): {raw[:500]}")
    print()


def test_chat_stream():
    print("=== POST /v1/chat/completions (stream) ===")
    payload = {
          "model": "mimo-v2.6-flash-free",
           "max_tokens": 32000,
           "reasoning_effort": "low",
           "messages": [
            {"role": "user", "content": "Count from 1 to 5."},
        ],
        "stream": True,
    }
    payload = json.load(open('opencode_request_body.json','r'))
   # payload['tools']=[]
    print(len(payload['tools']))
#    del payload['tools'][:]
    print(payload)
#    json.dump(payload,open('opencode_request_body.json','w'))
    url = f"{BASE_URL}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"Status: {resp.status}")
            print(f"Content-Type: {resp.headers.get('Content-Type')}")
            count = 0
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        print("[DONE]")
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        delta = choices[0].get("delta", {}) if choices else {}
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            count += 1
                    except (json.JSONDecodeError, IndexError):
                        pass
            print(f"\n\nStreamed {count} text chunks.")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_invalid_model():
    print("=== POST /v1/chat/completions (invalid model fallback) ===")
    payload = {
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "test"}],
        "stream": False,
        "max_tokens": 64,
    }
    payload = json.load(open('opencode_request_body.json','r'))
    status, raw = post("/v1/chat/completions", payload)
    if raw:
        try:
            data = json.loads(raw.decode("utf-8"))
            print(f"Status: {status}")
            print(f"Model used: {data.get('model')}")
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            print(f"Content: {msg.get('content', '')[:200]}")
        except json.JSONDecodeError:
            print(f"Status: {status}")
            print(f"Raw: {raw[:200]}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenCode Zen AI Gateway")
    parser.add_argument("--models", action="store_true", help="Test /v1/models")
    parser.add_argument("--chat", action="store_true", help="Test non-stream chat")
    parser.add_argument("--stream", action="store_true", help="Test stream chat")
    parser.add_argument("--fallback", action="store_true", help="Test invalid model fallback")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.all or args.models:
        test_models()
    if args.all or args.chat:
        test_chat_nonstream()
    if args.all or args.stream:
        test_chat_stream()
    if args.all or args.fallback:
        test_invalid_model()

    if not any([args.models, args.chat, args.stream, args.fallback, args.all]):
        print("No test selected. Use --all or one of: --models --chat --stream --fallback")
        print("Example: python test_gateway.py --all")
        sys.exit(1)
