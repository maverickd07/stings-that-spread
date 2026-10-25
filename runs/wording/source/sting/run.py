"""Command line runner. Every output file is written once and skipped on rerun, so a crashed run resumes.

  python -m sting.run models                                   # live model ids at the provider
  python -m sting.run snapshots --out runs/main --n 20         # base games paused at round 2 + predisposition
  python -m sting.run branches  --out runs/main --conditions N0,L0,L1,L2,L3,PC --after-rounds 3
  python -m sting.run branches  --out runs/main --conditions N0,L2,PC --after-rounds -1 --exit   # full games

Set STING_MOCK=1 to run everything against a fake model.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .game import CONDITIONS, Game
from .llm import LLM, ROSTERS, _all_models


def _models(arg: str) -> list:
    if arg in ROSTERS:
        return list(ROSTERS[arg])
    models = [m.strip() for m in arg.split(",") if m.strip()]
    if len(models) == 1:
        models *= 3
    assert len(models) == 3, "give a roster name, one model, or three models"
    return models


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _save_usage(out: Path, llm: LLM, tag: str, t0: float) -> None:
    usage = {k: dict(v) for k, v in llm.usage.items()}
    total = sum(v["calls"] for v in usage.values())
    _write(out / f"usage_{tag}_{int(t0)}.json", {"seconds": round(time.time() - t0), "provider": llm.provider,
                                                 "total_calls": total, "by_model": usage})
    print(f"[{tag}] {total} calls in {time.time() - t0:.0f}s: {json.dumps(usage)}")


def cmd_models(args) -> None:
    llm = LLM()
    live = set(llm.list_models())
    for short, ids in _all_models().items():
        mid = ids.get(llm.provider)
        if mid:
            print(f"{'OK ' if mid in live else 'MISSING'} {short:20s} {mid}")


def cmd_snapshots(args) -> None:
    out = Path(args.out)
    snap_dir = out / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    llm, models, t0 = LLM(), _models(args.models), time.time()
    config = {"models": models, "agent_mode": args.agent_mode, "branch_round": args.branch_round,
              "seed": args.seed, "pm_samples": args.pm_samples, "n": args.n,
              "provider": llm.provider, "hosting_provider": os.environ.get("STING_HOSTING_PROVIDER", llm.provider),
              "temperatures": {"play_challenge_offer_designer_exit": 0.7, "pm_tool": 1.0, "judge": 0.0},
              "game_seeds": list(range(args.seed, args.seed + args.n)),
              "sampling_seed": None, "mock": llm.mock}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing snapshot config differs; use a new output directory.")
    if not config_path.exists():
        _write(config_path, config)

    def one(i: int) -> str:
        path = snap_dir / f"snap_{i:03d}.json"
        if path.exists():
            return f"snap {i}: exists"
        game = Game(llm, models, seed=args.seed + i, agent_mode=args.agent_mode)
        status = game.run(stop_at_round=args.branch_round)
        if status != "paused" or not game.uc.alive:
            # The game ended or U died before the branch point: record and skip (counted in the analysis).
            _write(path, {"snap": i, "usable": False, "status": status, "uc_alive": game.uc.alive})
            return f"snap {i}: unusable ({status})"
        pre = game.predisposition(samples=args.pm_samples)
        with open(snap_dir / f"snap_{i:03d}.pkl", "wb") as f:
            pickle.dump(game.snapshot(), f)
        _write(path, {"snap": i, "usable": True, "seed": args.seed + i, "round": game.round,
                      "models": models, "agent_mode": args.agent_mode, "uc": game.uc.name,
                      "seats": [{"name": s.name, "model": s.model, "undercover": s.undercover, "alive": s.alive}
                                for s in game.seats],
                      "predisposition": pre, "fallbacks": game.fallbacks})
        return f"snap {i}: ok"

    _pool(one, range(args.n), args.workers)
    _save_usage(out, llm, "snapshots", t0)


def cmd_branches(args) -> None:
    if args.offers_only:
        from .wording import cmd_offers_only
        return cmd_offers_only(args)
    out = Path(args.out)
    snap_dir, br_dir = out / "snapshots", out / "branches"
    br_dir.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        assert c in CONDITIONS, f"unknown condition {c}; choose from {list(CONDITIONS)}"
    snaps = sorted(p for p in snap_dir.glob("snap_*.pkl"))
    if args.limit:
        snaps = snaps[: args.limit]
    llm, t0 = LLM(), time.time()
    tag = "full" if args.after_rounds < 0 else f"r{args.after_rounds}"
    _write(out / f"invocation_branches_{tag}_{time.time_ns()}.json", vars(args))

    def one(job) -> str:
        k, snap_path, cond = job
        path = br_dir / f"{snap_path.stem}__{cond}__{tag}.json"
        if path.exists():
            return f"{path.name}: exists"
        with open(snap_path, "rb") as f:
            game: Game = pickle.load(f)
        game.llm = llm
        game.apply_branch(cond, max_attempts=args.max_attempts)
        if args.after_rounds < 0:
            game.run()
        elif args.after_rounds > 0:
            game.run(until_round=game.branch_round + args.after_rounds - 1)
        if args.exit and game.over:
            game.run_exit()
        res = game.result()
        res["snap"] = snap_path.stem
        if args.contagion:
            res["contagion"] = _contagion(game, snaps, k, llm)
        if args.save_state:
            state_dir = out / "branch_states"
            state_dir.mkdir(exist_ok=True)
            tmp = state_dir / (path.stem + ".pkl.tmp")
            with tmp.open("wb") as f:
                pickle.dump(game.snapshot(), f)
            tmp.replace(state_dir / (path.stem + ".pkl"))
        _write(path, res)
        return f"{path.name}: ok"

    jobs = [(k, s, c) for k, s in enumerate(snaps) for c in conds]
    _pool(one, jobs, args.workers)
    _save_usage(out, llm, f"branches_{tag}", t0)


def cmd_contagion(args) -> None:
    """Probe saved branch memories without replaying or altering completed games."""
    out = Path(args.out)
    snaps = sorted((out / "snapshots").glob("snap_*.pkl"))
    indexes = {p.stem: i for i, p in enumerate(snaps)}
    dest = out / "contagion"
    dest.mkdir(exist_ok=True)
    llm, t0 = LLM(), time.time()

    def one(path):
        target = dest / (path.stem + ".json")
        if target.exists():
            return f"{target.name}: exists"
        with path.open("rb") as f:
            game = pickle.load(f)
        game.llm = llm
        snap = path.stem.split("__")[0]
        probes = _contagion(game, snaps, indexes[snap], llm)
        _write(target, {"snap": snap, "condition": game.condition, "tag": path.stem.split("__")[-1],
                        "source_branch": path.stem + ".json", "contagion": probes})
        return f"{target.name}: ok"

    _pool(one, sorted((out / "branch_states").glob("*.pkl")), args.workers)
    _save_usage(out, llm, "contagion", t0)


def _contagion(game: Game, snaps: list, k: int, llm: LLM) -> list:
    """Each agent carries this game's memory into the next snapshot's table (new seats and names) and gets the
    neutral private-message tool once. Recruiting there = the sting spread collusion."""
    if len(snaps) < 2:
        return []
    with open(snaps[(k + 1) % len(snaps)], "rb") as f:
        nxt: Game = pickle.load(f)
    nxt.llm = llm
    res = []
    for s in game.agents():
        host = next((h for h in nxt.agents() if h.model == s.model and h.alive), None)
        if host is None:
            continue
        probe = nxt.pm_probe(host, memory=game.memory_of(s), samples=1)[0]
        res.append({"agent": s.name, "model": s.model, "accepted": bool(game.accepted.get(s.name)),
                    "host_snap": snaps[(k + 1) % len(snaps)].stem, "host_name": host.name, **probe})
    return res


def _pool(fn, jobs, workers: int) -> None:
    jobs = list(jobs)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{n}/{len(jobs)}] {f.result()}", flush=True)
            except Exception as e:   # one bad job must not kill the run; rerun picks it up
                failures += 1
                print(f"[{n}/{len(jobs)}] FAILED: {e!r}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"{failures}/{len(jobs)} jobs failed; incomplete outputs must be resumed before analysis")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sting.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    s = sub.add_parser("snapshots")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--models", default="small", help="roster name, one model, or three comma-separated models")
    s.add_argument("--agent-mode", default="rules", choices=["rules", "values"])
    s.add_argument("--branch-round", type=int, default=2)
    s.add_argument("--pm-samples", type=int, default=3)
    s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--workers", type=int, default=8)
    b = sub.add_parser("branches")
    b.add_argument("--out", required=True)
    b.add_argument("--conditions", default="N0,L0,L1,L2,L3,PC")
    b.add_argument("--after-rounds", type=int, default=3, help="rounds to play after the branch; 0 = offer only; -1 = to the end")
    b.add_argument("--max-attempts", type=int, default=3)
    b.add_argument("--exit", action="store_true", help="exit interview (full games only)")
    b.add_argument("--contagion", action="store_true")
    b.add_argument("--save-state", action="store_true", help="save branch memory for a later contagion probe")
    b.add_argument("--limit", type=int, default=0, help="use only the first N snapshots")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--offers-only", action="store_true", help="One reply per live agent, no continuation or follow-ups; separate output")
    b.add_argument("--source-out", help="Read-only existing run containing saved snapshots (required for --offers-only)")
    c = sub.add_parser("contagion")
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    {"models": cmd_models, "snapshots": cmd_snapshots, "branches": cmd_branches,
     "contagion": cmd_contagion}[args.cmd](args)


if __name__ == "__main__":
    main()
