"""brief/llm.py: backend order, fallthrough, cache, log, subprocess plumbing. No real CLIs."""
from __future__ import annotations

import json
import sys

import pytest

from courtside.brief import llm


def fake(name, calls, reply="ok", fail=None):
    def fn(prompt, system, timeout):
        calls.append((name, prompt, system, timeout))
        if fail:
            raise llm.LLMError(fail)
        return reply
    return fn


def test_first_backend_wins_and_result_is_cached(data_dir):
    calls = []
    b = {"a": fake("a", calls, "first"), "b": fake("b", calls, "second")}
    assert llm.complete("hello", system="sys", backends=b) == "first"
    assert [c[0] for c in calls] == ["a"]
    # same request: served from disk, no new backend call, even via a different backend set
    assert llm.complete("hello", system="sys", backends={"z": fake("z", calls, "other")}) == "first"
    assert len(calls) == 1
    files = list((data_dir / "llm_cache").glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["text"] == "first" and rec["backend"] == "a"


def test_cache_key_depends_on_prompt_system_and_schema_not_backend():
    k = llm.cache_key("p", "s", None)
    assert k == llm.cache_key("p", "s", None)
    assert k != llm.cache_key("p2", "s", None)
    assert k != llm.cache_key("p", "s2", None)
    assert k != llm.cache_key("p", "s", {"type": "object"})
    assert len(k) == 64


def test_refresh_bypasses_cache_read_but_rewrites(data_dir):
    calls = []
    b = {"a": fake("a", calls, "v1")}
    llm.complete("q", backends=b)
    b2 = {"a": fake("a", calls, "v2")}
    assert llm.complete("q", backends=b2, refresh=True) == "v2"
    assert llm.complete("q", backends=b) == "v2"


def test_use_cache_false_never_touches_disk(data_dir):
    calls = []
    llm.complete("q", backends={"a": fake("a", calls)}, use_cache=False)
    llm.complete("q", backends={"a": fake("a", calls)}, use_cache=False)
    assert len(calls) == 2
    assert not (data_dir / "llm_cache").exists()


def test_falls_through_on_error_empty_and_unknown(data_dir):
    calls = []
    b = {
        "bad": fake("bad", calls, fail="Not logged in"),
        "empty": fake("empty", calls, reply="   "),
        "good": fake("good", calls, reply="fine"),
    }
    out = llm.complete("x", backends=b, order=["nope", "bad", "empty", "good"])
    assert out == "fine"
    assert [c[0] for c in calls] == ["bad", "empty", "good"]
    lines = [json.loads(l) for l in (data_dir / "logs" / "llm.log").read_text().splitlines()]
    assert [(l["backend"], l["ok"]) for l in lines if not l.get("cached")] == [
        ("bad", False), ("empty", False), ("good", True)]
    assert "Not logged in" in lines[0]["error"]


def test_all_backends_failed_raises_with_reasons_and_does_not_cache(data_dir):
    calls = []
    b = {"a": fake("a", calls, fail="boom"), "b": fake("b", calls, fail="timeout after 180s")}
    with pytest.raises(llm.AllBackendsFailed) as e:
        llm.complete("x", backends=b)
    assert set(e.value.errors) == {"a", "b"}
    assert not list((data_dir / "llm_cache").glob("*.json")) if (data_dir / "llm_cache").exists() else True


def test_env_order_and_monkeypatched_registry(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "BACKENDS", {"claude": fake("claude", calls, fail="down"),
                                           "codex": fake("codex", calls, "from codex")})
    monkeypatch.delenv("COURTSIDE_LLM", raising=False)
    assert llm.backend_order() == ["claude", "codex"]
    assert llm.complete("p1") == "from codex"
    assert [c[0] for c in calls] == ["claude", "codex"]
    calls.clear()
    monkeypatch.setenv("COURTSIDE_LLM", "codex")
    assert llm.complete("p2") == "from codex"
    assert [c[0] for c in calls] == ["codex"]


def test_json_schema_is_appended_to_prompt_for_every_backend():
    calls = []
    llm.complete("write", json_schema={"type": "object"}, backends={"a": fake("a", calls)})
    assert "JSON Schema" in calls[0][1] and '"type": "object"' in calls[0][1]


def test_claude_command_flags(monkeypatch):
    monkeypatch.delenv("COURTSIDE_CLAUDE_MODEL", raising=False)
    cmd = llm.claude_command("SYS")
    assert cmd[:4] == ["claude", "-p", "--output-format", "text"]
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--tools") + 1] == ""          # no tools at all
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    for flag in ("--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config"):
        assert flag in cmd
    monkeypatch.setenv("COURTSIDE_CLAUDE_MODEL", "opus")
    assert llm.claude_command(None)[llm.claude_command(None).index("--model") + 1] == "opus"


def test_codex_command_flags():
    cmd = llm.codex_command("/tmp/out.txt")
    assert cmd[:3] == ["codex", "exec", "--skip-git-repo-check"]
    assert cmd[cmd.index("-s") + 1] == "read-only" and "--ephemeral" in cmd
    assert cmd[-1] == "-" and cmd[cmd.index("-o") + 1] == "/tmp/out.txt"


def test_run_uses_temp_cwd_and_kills_on_timeout(tmp_path):
    rc, out, _ = llm._run([sys.executable, "-c", "import os,sys;print(os.getcwd());print(sys.stdin.read())"],
                          "hi", 20)
    assert rc == 0
    cwd, echoed = out.strip().split("\n")[:2]
    assert "courtside_llm_" in cwd and echoed == "hi"
    with pytest.raises(llm.LLMError, match="timeout"):
        llm._run([sys.executable, "-c", "import time;time.sleep(30)"], "", 1)
    with pytest.raises(llm.LLMError, match="not installed"):
        llm._run(["definitely-not-a-binary-xyz"], "", 5)


def test_check_text_rejects_auth_errors():
    with pytest.raises(llm.LLMError):
        llm._check_text("Not logged in - please run /login")
    with pytest.raises(llm.LLMError):
        llm._check_text("   ")
    assert llm._check_text("A real answer.") == "A real answer."
