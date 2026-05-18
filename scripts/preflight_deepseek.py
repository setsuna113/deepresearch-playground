"""L0 preflight probe — verify every external endpoint the studio path touches.

Five checks, each with loud failure mode (HTTP status + response body + request
URL on non-2xx, exception class on transport errors):

1. DeepSeek /chat/completions via httpx (matches what curl would do).
2. DeepSeek /chat/completions via the openai SDK (matches what the Studio path
   actually uses through ModelClient).
3. DeepSeek /embeddings probe — expected to return non-200; documents the gap
   the ReMe layer has to work around.
4. OpenAI /embeddings probe (only if OPENAI_API_KEY is in the env).
5. Local vLLM /v1/models on 127.0.0.1:8001 — for `co_schedule_v0`'s local roles.

Exits 0 iff checks (1)+(2)+(5) pass. (3) is informational; (4) is informational
and only runs when the key is present.

Usage:
    uv run --extra dev python scripts/preflight_deepseek.py

No project imports — only stdlib + httpx + openai + pyyaml.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import APIError, AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse `.env` without depending on python-dotenv."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _redact(s: str | None) -> str:
    if not s:
        return "<empty>"
    if len(s) <= 10:
        return s[:2] + "***"
    return s[:6] + "***" + s[-4:]


def _print_row(tag: str, name: str, detail: str) -> None:
    print(f"  [{tag:^4}] {name:<32} {detail}")


def _print_http_failure(name: str, url: str, status: int, body: str) -> None:
    print(f"  [FAIL] {name:<32} status={status}")
    print(f"         url    = {url}")
    print(f"         body[:500] = {body[:500]}")


async def probe_deepseek_chat_httpx(
    base_url: str, api_key: str, model: str
) -> bool:
    """Direct HTTP probe — surfaces wrong model_id / api key fast."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as c:
        try:
            r = await c.post(url, json=payload, headers=headers)
        except Exception as e:
            print(f"  [FAIL] chat.completions (httpx)   {type(e).__name__}: {e}")
            print(f"         url={url}")
            return False
    if r.status_code != 200:
        _print_http_failure("chat.completions (httpx)", url, r.status_code, r.text)
        return False
    data = r.json()
    usage = data.get("usage") or {}
    _print_row(
        "OK",
        "chat.completions (httpx)",
        f"model={model} prompt={usage.get('prompt_tokens', '?')} completion={usage.get('completion_tokens', '?')}",
    )
    return True


async def probe_deepseek_chat_openai(
    base_url: str, api_key: str, model: str
) -> bool:
    """Same probe via the OpenAI SDK path the Studio uses."""
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
    except APIError as e:
        print(f"  [FAIL] chat.completions (openai)  APIError status={getattr(e, 'status_code', '?')}")
        print(f"         body = {getattr(e, 'body', None) or str(e)}")
        return False
    except Exception as e:
        print(f"  [FAIL] chat.completions (openai)  {type(e).__name__}: {e}")
        return False
    usage = resp.usage
    _print_row(
        "OK",
        "chat.completions (openai)",
        f"prompt={usage.prompt_tokens if usage else '?'} completion={usage.completion_tokens if usage else '?'}",
    )
    return True


async def probe_embeddings(name: str, base_url: str, api_key: str, model: str) -> bool:
    """Probe POST /embeddings. Returns True iff status==200."""
    url = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model, "input": "ping"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as c:
        try:
            r = await c.post(url, json=payload, headers=headers)
        except Exception as e:
            print(f"  [INFO] {name:<32} {type(e).__name__}: {e}")
            return False
    if r.status_code != 200:
        body = r.text[:300]
        _print_row("INFO", name, f"status={r.status_code} body={body!r}")
        return False
    try:
        dim = len(r.json()["data"][0]["embedding"])
    except Exception:
        dim = "?"
    _print_row("OK", name, f"status=200 dim={dim}")
    return True


async def probe_local_vllm(host: str = "127.0.0.1", port: int = 8001) -> bool:
    url = f"http://{host}:{port}/v1/models"
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(url)
        except httpx.ConnectError as e:
            _print_row("FAIL", "local vLLM /v1/models", f"ConnectError: {e} (run `bash scripts/serve_local.sh`)")
            return False
        except Exception as e:
            _print_row("FAIL", "local vLLM /v1/models", f"{type(e).__name__}: {e}")
            return False
    if r.status_code != 200:
        _print_http_failure("local vLLM /v1/models", url, r.status_code, r.text)
        return False
    try:
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        ids = []
    _print_row("OK", "local vLLM /v1/models", f"models={ids}")
    return True


async def main() -> int:
    env = {**_load_env_file(ROOT / ".env"), **os.environ}
    cfg = _load_yaml(ROOT / "config" / "config.local.yaml")

    cloud = cfg["models"]["endpoints"]["cloud"]
    cloud_base = cloud["base_url"]
    cloud_model = cloud["model_id"]

    # crude ${VAR:-default} resolution for `cloud.api_key` (e.g. "${CLOUD_API_KEY:-EMPTY}")
    cloud_key = env.get("CLOUD_API_KEY") or env.get("DEEPSEEK_API_KEY") or "EMPTY"

    print(f"=== Preflight ({cloud_base}, model={cloud_model}, key={_redact(cloud_key)}) ===")

    results: dict[str, bool] = {}
    results["chat_httpx"] = await probe_deepseek_chat_httpx(cloud_base, cloud_key, cloud_model)
    results["chat_openai"] = await probe_deepseek_chat_openai(cloud_base, cloud_key, cloud_model)
    await probe_embeddings(
        "embeddings on DeepSeek", cloud_base, cloud_key, "text-embedding-3-small"
    )

    openai_key = env.get("OPENAI_API_KEY")
    if openai_key:
        await probe_embeddings(
            "embeddings on OpenAI",
            "https://api.openai.com/v1",
            openai_key,
            "text-embedding-3-small",
        )
    else:
        _print_row("INFO", "embeddings on OpenAI", "OPENAI_API_KEY not set (skipped)")

    # ReMe embedding endpoint — if REME_EMBEDDING_API_BASE/KEY are set
    # (or DEEPSEEK fallback chain), probe with the configured model_id.
    reme_emb_base = env.get("REME_EMBEDDING_API_BASE")
    reme_emb_key = env.get("REME_EMBEDDING_API_KEY")
    reme_emb_model = (
        cfg.get("memory", {}).get("reme", {}).get("embedding", {}).get("model_id")
        or "text-embedding-3-small"
    )
    if reme_emb_base and reme_emb_key:
        results["reme_embeddings"] = await probe_embeddings(
            f"ReMe embeddings ({reme_emb_base.split('//')[-1].split('/')[0]})",
            reme_emb_base,
            reme_emb_key,
            reme_emb_model,
        )
    else:
        _print_row(
            "INFO",
            "ReMe embeddings endpoint",
            "REME_EMBEDDING_API_BASE/KEY not set (ReMe summary writes will fail if enabled)",
        )

    results["local_vllm"] = await probe_local_vllm()

    required = {"chat_httpx", "chat_openai", "local_vllm"}
    failed = [k for k in required if not results.get(k)]
    print()
    if failed:
        print(f"  RESULT: FAIL  (required probes failed: {failed})")
        return 1
    print("  RESULT: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
