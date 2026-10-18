"""Prose from the user's existing subscriptions via headless CLIs. No API keys.

    complete(prompt, system=..., json_schema=..., timeout=180) -> str

Backends are tried in the order given by env ``COURTSIDE_LLM`` (default ``"claude,codex"``);
a backend that errors, times out, is not logged in, or returns nothing falls through to the
next. Results are cached on disk (``data_dir()/llm_cache/<sha256>.json``) keyed by the
backend-independent ``prompt + system + schema``, so rebuilding a Brief costs no usage.
Every call (including cache hits and failures) is appended to ``data_dir()/logs/llm.log``.

Both CLIs run from a throwaway temp directory so they can never touch the repo.

Env knobs: ``COURTSIDE_LLM``, ``COURTSIDE_CLAUDE_MODEL`` (default ``sonnet``),
``COURTSIDE_CLAUDE_EFFORT`` (unset = CLI default), ``COURTSIDE_CODEX_MODEL`` (unset = codex
default), ``COURTSIDE_LLM_REFRESH=1`` (ignore the cache on read; still writes), ``COURTSIDE_NO_WRITER=1``
(never launch the real claude/codex CLIs; cache hits and injected fakes still work).

Tests inject fakes either with ``complete(..., backends={"fake": fn})`` (``fn(prompt, system,
timeout) -> str``) or by monkeypatching ``llm.BACKENDS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from courtside.config import data_dir

Backend = Callable[[str, "str | None", int], str]

DEFAULT_ORDER = "claude,codex"
DEFAULT_CLAUDE_MODEL = "sonnet"
DEFAULT_SYSTEM = (
    "You are a precise sports copywriter. Follow the output format exactly and output "
    "nothing else."
)
# Output that is really an error message from the CLI (it can exit 0 with these).
_AUTH_MARKERS = (
    "not logged in", "please run /login", "invalid api key", "authentication_error",
    "401 unauthorized", "token has expired", "you've hit your", "usage limit",
    "rate limit", "credit balance",
)


class LLMError(RuntimeError):
    """One backend failed (exit code, auth, timeout, empty output)."""


class AllBackendsFailed(LLMError):
    """Every configured backend failed; ``errors`` maps backend name -> message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("all LLM backends failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))


# ---------------------------------------------------------------------------------------
# subprocess plumbing
# ---------------------------------------------------------------------------------------

def _run(cmd: list[str], stdin: str, timeout: int) -> tuple[int, str, str]:
    """Run ``cmd`` in a temp cwd with a hard timeout (kills the whole process group)."""
    with tempfile.TemporaryDirectory(prefix="courtside_llm_") as cwd:
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True,
            )
        except FileNotFoundError as e:
            raise LLMError(f"{cmd[0]} not installed") from e
        try:
            out, err = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            raise LLMError(f"timeout after {timeout}s") from e
        return proc.returncode, out, err


def _check_text(text: str, err: str = "") -> str:
    text = (text or "").strip()
    if not text:
        raise LLMError("empty output" + (f" ({err.strip()[:200]})" if err.strip() else ""))
    low = text[:400].lower()
    if len(text) < 400 and any(m in low for m in _AUTH_MARKERS):
        raise LLMError(f"auth/limit error: {text[:200]}")
    return text


# ---------------------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------------------

def claude_command(system: str | None) -> list[str]:
    """``claude -p`` with no tools, no persisted session, no user settings/skills/MCP.

    * ``--tools ""``                 text only: the model can't read or edit anything
    * ``--no-session-persistence``   don't litter ~/.claude with one-shot sessions
    * ``--setting-sources ""``       ignore user/project settings, hooks, CLAUDE.md memory
    * ``--disable-slash-commands``   no skills; ``--strict-mcp-config`` no MCP servers
    * ``--system-prompt``            replaces Claude Code's agentic system prompt
    """
    cmd = [
        "claude", "-p", "--output-format", "text",
        "--model", os.environ.get("COURTSIDE_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
        "--tools", "", "--no-session-persistence", "--setting-sources", "",
        "--disable-slash-commands", "--strict-mcp-config",
        "--system-prompt", system or DEFAULT_SYSTEM,
    ]
    effort = os.environ.get("COURTSIDE_CLAUDE_EFFORT")
    if effort:
        cmd += ["--effort", effort]
    return cmd


def claude_backend(prompt: str, system: str | None, timeout: int) -> str:
    rc, out, err = _run(claude_command(system), prompt, timeout)
    if rc != 0:
        raise LLMError(f"claude exited {rc}: {(err or out).strip()[:300]}")
    return _check_text(out, err)


def codex_command(out_file: str) -> list[str]:
    """``codex exec`` read-only, ephemeral, prompt on stdin (``-``), final message to a file."""
    cmd = [
        "codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral",
        "--color", "never", "-o", out_file,
    ]
    model = os.environ.get("COURTSIDE_CODEX_MODEL")
    if model:
        cmd += ["-m", model]
    return cmd + ["-"]


def codex_backend(prompt: str, system: str | None, timeout: int) -> str:
    # codex has no system-prompt flag: fold the system text into the prompt.
    full = f"{system}\n\n---\n\n{prompt}" if system else f"{DEFAULT_SYSTEM}\n\n---\n\n{prompt}"
    with tempfile.TemporaryDirectory(prefix="courtside_codex_") as d:
        out_file = str(Path(d) / "last_message.txt")
        rc, out, err = _run(codex_command(out_file), full, timeout)
        if rc != 0:
            raise LLMError(f"codex exited {rc}: {(err or out).strip()[:300]}")
        try:
            text = Path(out_file).read_text()
        except OSError:
            text = ""
        return _check_text(text, err)


BACKENDS: dict[str, Backend] = {"claude": claude_backend, "codex": codex_backend}


def backend_order() -> list[str]:
    raw = os.environ.get("COURTSIDE_LLM") or DEFAULT_ORDER
    return [b.strip().lower() for b in raw.split(",") if b.strip()]


# ---------------------------------------------------------------------------------------
# cache + log
# ---------------------------------------------------------------------------------------

def cache_key(prompt: str, system: str | None = None, json_schema: dict | None = None) -> str:
    """Backend-independent: the same request hits the cache whichever CLI answered first."""
    blob = json.dumps({"prompt": prompt, "system": system or "", "schema": json_schema or None},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_dir() -> Path:
    return data_dir() / "llm_cache"


def _cache_read(key: str) -> dict | None:
    p = cache_dir() / f"{key}.json"
    try:
        rec = json.loads(p.read_text())
        return rec if isinstance(rec.get("text"), str) and rec["text"].strip() else None
    except (OSError, ValueError):
        return None


def _cache_write(key: str, text: str, backend: str, seconds: float) -> None:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    rec = {"text": text, "backend": backend, "seconds": round(seconds, 2),
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = d / f".{key}.tmp"
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.replace(d / f"{key}.json")


def log_path() -> Path:
    return data_dir() / "logs" / "llm.log"


def _log(**rec) -> None:
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
        with p.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break a build


# ---------------------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------------------

def _with_schema(prompt: str, json_schema: dict | None) -> str:
    if not json_schema:
        return prompt
    return (prompt + "\n\nRespond with ONLY a JSON document (no code fences, no commentary) "
            "that validates against this JSON Schema:\n"
            + json.dumps(json_schema, ensure_ascii=False))


def complete(
    prompt: str,
    *,
    system: str | None = None,
    json_schema: dict | None = None,
    timeout: int = 180,
    backends: Mapping[str, Backend] | None = None,
    order: list[str] | None = None,
    use_cache: bool = True,
    refresh: bool | None = None,
) -> str:
    """Return the model's text for ``prompt``. Raises ``AllBackendsFailed`` if none worked.

    ``json_schema`` is appended to the prompt as an instruction (the CLIs' own structured
    output modes differ and aren't needed); callers still parse/validate the reply.
    ``backends`` / ``order`` override the registry and env for tests.
    """
    key = cache_key(prompt, system, json_schema)
    if refresh is None:
        refresh = os.environ.get("COURTSIDE_LLM_REFRESH") == "1"
    if use_cache and not refresh:
        hit = _cache_read(key)
        if hit:
            _log(backend=hit.get("backend"), cached=True, ok=True, seconds=0.0,
                 chars_in=len(prompt), chars_out=len(hit["text"]), key=key[:12])
            return hit["text"]

    registry = BACKENDS if backends is None else backends
    names = order or (list(registry) if backends is not None else backend_order())
    full_prompt = _with_schema(prompt, json_schema)
    errors: dict[str, str] = {}
    for name in names:
        fn = registry.get(name)
        if fn is None:
            errors[name] = "unknown backend"
            continue
        if fn in (claude_backend, codex_backend) and os.environ.get("COURTSIDE_NO_WRITER"):
            errors[name] = "disabled by COURTSIDE_NO_WRITER"   # real CLIs only; fakes still run
            continue
        t0 = time.monotonic()
        try:
            text = fn(full_prompt, system, timeout)
            if not isinstance(text, str) or not text.strip():
                raise LLMError("empty output")
        except Exception as e:  # noqa: BLE001 - any backend failure falls through
            dt = time.monotonic() - t0
            errors[name] = f"{type(e).__name__}: {e}"[:300]
            _log(backend=name, cached=False, ok=False, seconds=round(dt, 2),
                 chars_in=len(full_prompt), error=errors[name])
            continue
        dt = time.monotonic() - t0
        _log(backend=name, cached=False, ok=True, seconds=round(dt, 2),
             chars_in=len(full_prompt), chars_out=len(text), key=key[:12])
        if use_cache:
            _cache_write(key, text, name, dt)
        return text
    raise AllBackendsFailed(errors)
