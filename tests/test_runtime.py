"""Regression checks for failures that can corrupt or stop a real screening run."""
import io
import json
import urllib.error
import hashlib
import pickle

import pandas as pd
import pytest

from sting.analyze import entrapment_table
from sting.budget import rate, reported_usage
from sting.llm import LLM
from sting.run import _pool


def test_failed_http_requests_are_logged_and_not_counted_as_refusals(monkeypatch, tmp_path):
    monkeypatch.setenv("STING_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("STING_API_KEY", "private-test-key")
    raw = tmp_path / "raw.jsonl"
    monkeypatch.setenv("STING_RAW_LOG", str(raw))

    def fail(*a, **kw):
        raise urllib.error.HTTPError("https://example.invalid", 503, "unavailable", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="failed after retries"):
        LLM(mock=False).chat("test/model", [{"role": "user", "content": "test"}])
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(records) == 5
    assert all(r["http_status"] == 503 for r in records)
    assert "private-test-key" not in raw.read_text()


def test_worker_failure_propagates_to_the_command():
    def fail(_):
        raise ValueError("broken snapshot")

    with pytest.raises(RuntimeError, match="1/1 jobs failed"):
        _pool(fail, [0], workers=1)


def test_induced_share_is_acceptance_minus_spontaneous():
    base = {"run": "test", "snap": "snap_000", "model": "model"}
    pred = pd.DataFrame([{**base, "agent": str(i), "pm_any": i == 0, "designer_accept": False}
                         for i in range(4)])
    offers = pd.DataFrame([{**base, "agent": str(i), "tag": "r0", "condition": "L2",
                            "accept1": i < 2, "accepted_at": 1 if i < 2 else None,
                            "suspicion1": False} for i in range(4)])
    row = entrapment_table(offers, pred).iloc[0]
    assert row["accept"] == 0.5
    assert row["spontaneous"] == 0.25
    assert row["induced_share"] == 0.25
    assert row["acceptor_nonspontaneous_share"] == 0.5
    assert row["precision"] == 0.5
    assert row["objective_entrapment"] == pytest.approx(1 / 3)


def test_multi_gpu_budget_counts_both_gpus():
    assert rate("H100:2") - rate("H100") == pytest.approx(0.001097)
    assert reported_usage({"metered_cost": "0.0002", "metered_cost_breakdown": {
        "ephemeral_apps": "0.0039", "network_egress": "0.0002"}}) == pytest.approx(0.0041)


def test_deferred_contagion_preserves_completed_games_and_memory(monkeypatch, tmp_path):
    from sting.run import main
    from sting.analyze import load
    from sting.judge import collect
    monkeypatch.setenv("STING_MOCK", "1")
    main(["snapshots", "--out", str(tmp_path), "--n", "2", "--seed", "3", "--models", "test/model",
          "--workers", "1"])
    main(["branches", "--out", str(tmp_path), "--conditions", "N0,L2", "--after-rounds", "3",
          "--save-state", "--workers", "1"])
    paths = list((tmp_path / "branches").glob("*.json"))
    assert len(paths) == 4
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    memories = []
    for p in (tmp_path / "branch_states").glob("*.pkl"):
        with p.open("rb") as f:
            game = pickle.load(f)
        assert game.llm is None
        memories.extend(game.memory_of(s) for s in game.agents())
    assert any("Private messages you exchanged" in m for m in memories)
    main(["contagion", "--out", str(tmp_path), "--workers", "1"])
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    probes = [json.loads(p.read_text()) for p in (tmp_path / "contagion").glob("*.json")]
    expected = sum(len(p["contagion"]) for p in probes)
    assert expected > 0
    assert len(load(tmp_path)["contagion"]) == expected
    assert len(list(collect(tmp_path))) > 0
    saved = {p: p.read_bytes() for p in (tmp_path / "contagion").glob("*.json")}
    main(["contagion", "--out", str(tmp_path), "--workers", "1"])
    assert saved == {p: p.read_bytes() for p in saved}


def test_judge_invalid_json_retry_constrains_format_without_changing_prompt(monkeypatch, tmp_path):
    import sting.judge as judge
    calls = []

    class FakeLLM:
        provider = "custom"
        mock = False
        usage = {}

        def chat(self, model, messages, **kwargs):
            calls.append((messages, kwargs))
            return '{"suspicion": false, ""}' if kwargs['response_format'] is None else '{"suspicion": false, "evidence": ""}'

    monkeypatch.setattr(judge, "LLM", FakeLLM)
    monkeypatch.setattr(judge, "collect", lambda out: [("failed-item", "suspicion", {"offer": "offer", "reply": "reply"})])
    judge.main(["--out", str(tmp_path), "--model", "test/model", "--workers", "1"])
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert calls[0][1]["temperature"] == calls[1][1]["temperature"] == 0
    assert calls[1][1]["response_format"] == {"type": "json_object"}
    result = json.loads((tmp_path / "judgments.json").read_text())["failed-item"]
    assert result["judge"]["suspicion"] is False
    assert result["response_format"] == {"type": "json_object"}


def test_json_parser_ignores_braces_inside_quoted_messages():
    from sting.llm import parse_json
    text = '```json\n{"to": "Lily", "message": "truce if you have the Queen in your hand}"}\n```'
    assert parse_json(text) == {"to": "Lily", "message": "truce if you have the Queen in your hand}"}
    assert parse_json('{"to": "Lily", "message": "unterminated}') is None
