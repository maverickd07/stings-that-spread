"""Offline checks of the engine against the mock model: run with `STING_MOCK=1 pytest -q`."""
import random

import pytest

from sting import prompts as P
from sting.game import CONDITIONS, Game, parse_accept
from sting.llm import LLM, parse_json

MODELS = ["llama-3.1-8b", "qwen2.5-7b", "ministral-8b"]


@pytest.fixture
def llm():
    random.seed(0)
    return LLM(mock=True)


class Spy(LLM):
    """Mock that records every prompt."""

    def __init__(self):
        super().__init__(mock=True)
        self.prompts = []

    def chat(self, model, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return super().chat(model, messages, **kw)


def test_full_game_ends_and_cards_are_legal(llm):
    for seed in range(30):
        g = Game(llm, MODELS, seed=seed)
        assert g.run() == "over"
        assert len(g.alive()) <= 1 or g.round == g.max_rounds
        for d in g.decisions:
            if d["type"] == "play":
                assert 1 <= d["n"] <= 3


def test_one_undercover_and_shuffled_seats(llm):
    orders = set()
    for seed in range(20):
        g = Game(llm, MODELS, seed=seed)
        assert sum(s.undercover for s in g.seats) == 1
        assert sorted(s.model for s in g.agents()) == sorted(MODELS)
        orders.add(tuple(s.name for s in g.seats))
    assert len(orders) > 5


def test_pause_snapshot_and_branch_are_independent(llm):
    g = Game(llm, MODELS, seed=3)
    assert g.run(stop_at_round=2) == "paused" or not g.uc.alive
    snap = g.snapshot()
    a, b = g.snapshot(), g.snapshot()
    a.llm = b.llm = llm
    a.apply_branch("L2")
    b.apply_branch("N0")
    assert a.offers and not b.offers
    assert all(not s.private for s in b.agents())
    assert all(not s.private for s in snap.agents())   # branching never leaks back


def test_repeated_offers_stop_after_accept_or_cap(llm):
    for seed in range(15):
        g = Game(llm, MODELS, seed=seed)
        if g.run(stop_at_round=2) != "paused" or not g.uc.alive:
            continue
        g.apply_branch("L2", max_attempts=3)
        g.run()
        per_agent = {}
        for o in g.offers:
            per_agent.setdefault(o["agent"], []).append(o)
        for offers in per_agent.values():
            assert len(offers) <= 3
            assert [o["attempt"] for o in offers] == list(range(1, len(offers) + 1))
            assert all(not o["accept"] for o in offers[:-1])   # nobody is asked again after accepting


def test_pc_reminder_only_in_positive_control():
    for cond in ("PC", "L2", "N0"):
        spy = Spy()
        random.seed(1)
        g = Game(spy, MODELS, seed=11)
        if g.run(stop_at_round=2) != "paused" or not g.uc.alive:
            pytest.skip("undercover died early for this seed")
        g.apply_branch(cond)
        g.run()
        has = any("DO NOT CHALLENGE" in p for p in spy.prompts)
        llm_challenges_on_uc = any(d["type"] == "challenge" and d["target_uc"] and not d["actor_uc"]
                                   and d["phase"] == "post" for d in g.decisions)
        assert has == (cond == "PC" and llm_challenges_on_uc)


def test_probes_are_not_stored(llm):
    g = Game(llm, MODELS, seed=5)
    g.run(stop_at_round=2)
    before = [list(s.private) for s in g.seats]
    g.predisposition(samples=2)
    assert [list(s.private) for s in g.seats] == before


def test_deterrence_notice_reaches_prompts():
    spy = Spy()
    g = Game(spy, MODELS, seed=7)
    g.run(stop_at_round=2)
    if not g.uc.alive:
        pytest.skip("undercover died early")
    g.apply_branch("D50")
    assert any("50% of private deal offers" in p for p in spy.prompts if "TASK: OFFER_REPLY" in p)


def test_parsers():
    assert parse_accept("ACCEPT - PARTNER: lily - ok") == {"accept": True, "partner": "Lily"}
    assert parse_accept("I refuse, I won't ACCEPT this")["accept"] is False
    assert parse_accept("Sure!")["accept"] is False
    assert parse_json('```json\n{"a": {"b": 1}}\n``` trailing') == {"a": {"b": 1}}
    assert parse_json("no json") is None


def test_all_conditions_run(llm):
    g = Game(llm, MODELS, seed=2)
    g.run(stop_at_round=2)
    for c in CONDITIONS:
        b = g.snapshot()
        b.llm = llm
        b.apply_branch(c)
        b.run(until_round=b.branch_round + 2)
    assert P.OFFERS.keys() >= {"L0", "L1", "L2", "L3"}
