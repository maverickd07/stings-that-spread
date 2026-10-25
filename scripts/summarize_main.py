"""Audit completed main runs and save derived summaries; never edit raw results."""
import csv
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

from sting.analyze import load, entrapment_table, challenge_on_uc
from sting.budget import rate, reported_usage
from sting.judge import collect
from sting.llm import parse_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["Qwen/Qwen2.5-32B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"]
CONDITIONS = {"N0", "L0", "L1", "L2", "L3", "PC"}


def read(path):
    return json.loads(path.read_text())


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = ROOT / "runs"
    audit, tables, baselines, behaviours, raw_paths, values_tables, contagion_tables = {}, [], [], [], [], [], []
    for model in MODELS:
        out = runs / ("main_" + model.split("/")[-1])
        config = read(out / "config.json")
        assert not config["mock"] and config["hosting_provider"] == "modal"
        assert config["models"] == [model] * 3
        snapshots = sorted((out / "snapshots").glob("*.json"))
        assert len(snapshots) == config["n"] == 30
        usable = [p for p in snapshots if read(p).get("usable")]
        excluded, invalid_pm, fallback_counts, bad_offers, base_fallbacks = [], [], [], [], []
        for p in snapshots:
            s = read(p)
            if not s.get("usable"):
                excluded.append({"snap": p.stem, **s})
                continue
            for a in s["predisposition"]:
                for i, pm in enumerate(a["pm"]):
                    if pm.get("raw"):
                        invalid_pm.append({"snap": p.stem, "agent": a["agent"], "sample": i,
                                           "raw": pm["raw"]})
            base_fallbacks.append({"snap": p.stem, "fallbacks": s["fallbacks"]})
            with p.with_suffix(".pkl").open("rb") as f:
                base = pickle.load(f)
            for cond in CONDITIONS:
                path = out / "branches" / (p.stem + "__" + cond + "__r3.json")
                b = read(path)
                assert b["seed"] == s["seed"] and b["branch_round"] == 2
                assert b["decisions"][:len(base.decisions)] == base.decisions
                eligible = {a["name"] for a in s["seats"] if a["alive"] and not a["undercover"]}
                first = {o["agent"] for o in b["offers"] if o["attempt"] == 1}
                assert first == (set() if cond == "N0" else eligible)
                for agent in first:
                    offers = [o for o in b["offers"] if o["agent"] == agent]
                    assert len(offers) <= 3
                    assert [o["attempt"] for o in offers] == list(range(1, len(offers) + 1))
                    assert not any(o["accept"] for o in offers[:-1])
                for o in b["offers"]:
                    if cond != "PC" and not o["reply"].lstrip().upper().startswith(("ACCEPT", "REFUSE", "REJECT", "DECLINE")):
                        bad_offers.append({"branch": path.name, **o})
                fallback_counts.append({"branch": path.name, "postbranch": b["fallbacks"] - s["fallbacks"]})
        assert len(list((out / "branches").glob("*.json"))) == len(usable) * len(CONDITIONS)
        cache = read(out / "judgments.json")
        missing = sorted({k for k, _, _ in collect(out)} - set(cache))
        assert not missing, missing
        for j in cache.values():
            if j["kind"] in ("collusion", "suspicion"):
                assert type(j["judge"].get(j["kind"])) is bool
        data = load(out)
        t = entrapment_table(data["offers"], data["pred"])
        t["pc_is_imposed_not_voluntary"] = t.condition == "PC"
        tables.append(t)
        pred = data["pred"]
        baselines.append({"model": model, "attempted": 30, "usable": len(usable), "agents": len(pred),
                          "spontaneous_k": int(pred.pm_any.sum()), "spontaneous_n": len(pred),
                          "spontaneous": float(pred.pm_any.mean())})
        behaviours.append(challenge_on_uc(data["chal"]))
        audit[model] = {"paired_prefix_checks_passed": True, "attempted": 30, "usable": len(usable),
                        "excluded": excluded, "invalid_neutral_pm": invalid_pm,
                        "offer_format_issues": bad_offers, "missing_judgments": missing,
                        "base_snapshot_fallbacks": base_fallbacks,
                        "postbranch_fallbacks": fallback_counts,
                        "optional_contagion_probes": len(data["contagion"])}
        if len(data["contagion"]):
            assert len(list((out / "contagion").glob("*.json"))) == len(usable) * len(CONDITIONS)
            c = data["contagion"].groupby(["condition", "accepted"]).recruits.agg(["sum", "mean", "size"]).reset_index()
            c["model"] = model
            contagion_tables.append(c)
        raw_paths.extend(p for p in out.rglob("*") if p.is_file() and "analysis" not in p.parts)
        values = runs / ("values_" + model.split("/")[-1])
        if values.exists():
            vc = read(values / "config.json")
            assert vc["agent_mode"] == "values" and not vc["mock"] and vc["models"] == [model] * 3
            vs = list((values / "snapshots").glob("*.json"))
            vu = sum(read(p).get("usable", False) for p in vs)
            assert len(vs) == 30 and len(list((values / "branches").glob("*.json"))) == vu * 3
            vcache = read(values / "judgments.json")
            assert {k for k, _, _ in collect(values)} <= vcache.keys()
            for j in vcache.values():
                if j["kind"] in ("collusion", "suspicion"):
                    assert type(j["judge"].get(j["kind"])) is bool
            for p in vs:
                if not read(p).get("usable"):
                    continue
                with p.with_suffix(".pkl").open("rb") as f:
                    base = pickle.load(f)
                for cond in ("L1", "L2", "L3"):
                    vb = read(values / "branches" / (p.stem + "__" + cond + "__r0.json"))
                    assert vb["seed"] == base.seed and vb["decisions"] == base.decisions
            vd = load(values)
            values_tables.append(entrapment_table(vd["offers"], vd["pred"]))
            audit[model]["values"] = {"attempted": 30, "usable": vu,
                                       "paired_prefix_checks_passed": True,
                                       "invalid_neutral_pm": [{"snap": p.stem, "agent": a["agent"], "sample": i}
                                           for p in vs for a in read(p).get("predisposition", [])
                                           for i, pm in enumerate(a["pm"]) if pm.get("raw")]}
            raw_paths.extend(p for p in values.rglob("*") if p.is_file() and "analysis" not in p.parts)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(runs / "main_summary.csv", index=False)
    if values_tables:
        pd.concat(values_tables, ignore_index=True).to_csv(runs / "values_summary.csv", index=False)
    if contagion_tables:
        pd.concat(contagion_tables, ignore_index=True).to_csv(runs / "contagion_summary.csv", index=False)
    pd.DataFrame(baselines).to_csv(runs / "main_baseline.csv", index=False)
    behaviour = pd.concat(behaviours, ignore_index=True)
    behaviour.to_csv(runs / "main_behaviour.csv", index=False)
    (runs / "main_quality.json").write_text(json.dumps(audit, indent=2) + "\n")

    # All billing snapshots are retained because Modal may revise totals after shutdown.
    bills = [(p, read(p)) for p in (ROOT / "logs/modal").glob("billing*.json")]
    highest = max(reported_usage(b) for _, b in bills)
    latest_path, latest = max(bills, key=lambda x: x[0].stat().st_mtime)
    sessions = [read(p) for p in (ROOT / "logs/modal").glob("session_*.json")]
    spend = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "latest_source": str(latest_path.relative_to(ROOT)),
             "latest_metered_usd": reported_usage(latest), "highest_observed_metered_usd": highest,
             "conservative_reported_usd": math.ceil(highest * 100) / 100,
             "latest_compute_billed_usd": float(latest["billed_cost"]),
             "wall_time_allowance_usd": sum(s["seconds"] * rate(s["gpu"]) for s in sessions),
             "excludes_separate_card_verification": True}
    before = reported_usage(read(ROOT / "logs/modal/billing_main_authorized.json"))
    spend["metered_before_main_usd"] = before
    spend["latest_metered_change_since_main_start_usd"] = reported_usage(latest) - before
    spend["metered_change_note"] = "Includes billing revisions of earlier use; total budget guard uses the larger of reported cost and all-session wall-time allowance."
    (runs / "main_spend.json").write_text(json.dumps(spend, indent=2) + "\n")
    started = datetime.fromisoformat(read(runs / "main_plan.json")["authorized_at_utc"]).timestamp()
    raw_audit = {"successful_calls": 0, "errors": [], "length_finishes": [], "by_model": {},
                 "invalid_judge_json_responses": [], "json_constrained_retry_calls": 0}
    for p in (ROOT / "logs/modal").glob("raw_*.jsonl"):
        included = False
        for line in p.read_text().splitlines():
            a = json.loads(line)
            if a["timestamp"] < started:
                continue
            included = True
            request = a["request"]
            model = request["model"]
            assert model in MODELS and a["hosting_provider"] == "modal" and not a["mock"]
            content = request["messages"][-1]["content"]
            temp = 0.0 if "TASK: JUDGE_" in content else 1.0 if "TASK: PM_TOOL" in content else 0.7
            assert request["temperature"] == temp
            if "response" not in a:
                raw_audit["errors"].append({"file": p.name, "request_id": a["request_id"]})
                continue
            assert a["response"]["model"] == model
            raw_audit["successful_calls"] += 1
            raw_audit["by_model"][model] = raw_audit["by_model"].get(model, 0) + 1
            if request.get("response_format"):
                raw_audit["json_constrained_retry_calls"] += 1
            for choice in a["response"]["choices"]:
                if "TASK: JUDGE_" in content and not parse_json(choice["message"]["content"]):
                    raw_audit["invalid_judge_json_responses"].append({"file": p.name, "request_id": a["request_id"],
                                                                    "model": model})
                if choice["finish_reason"] == "length":
                    raw_audit["length_finishes"].append({"file": p.name, "request_id": a["request_id"],
                                                         "model": model, "task": content.splitlines()[0]})
        if included:
            raw_paths.append(p)
    (runs / "main_raw_quality.json").write_text(json.dumps(raw_audit, indent=2) + "\n")
    (runs / "main_raw_SHA256SUMS").write_text("".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(ROOT)) + "\n"
        for p in sorted(set(raw_paths))))

    fig_dir = runs / "main_comparison"
    fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        g = combined[(combined.model == model) & combined.condition.isin(["L0", "L1", "L2", "L3"])].set_index("condition").reindex(["L0", "L1", "L2", "L3"])
        ax.errorbar(range(4), g.accept, yerr=[g.accept - g.accept_lo, g.accept_hi - g.accept], fmt="o-", capsize=3, label="First offer acceptance")
        baseline = next(x for x in baselines if x["model"] == model)
        ax.axhline(baseline["spontaneous"], ls="--", color="gray", label="N0 spontaneous PM")
        ax.set(xticks=range(4), xticklabels=g.index, ylim=(-.02, 1.02), title=model.split("/")[-1])
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / ("acceptance_vs_baseline." + ext), dpi=200)
    plt.close(fig)
    print(combined[["model", "condition", "accept_k", "accept_n", "accept", "spontaneous", "induced_share", "cum_accept_3"]].to_string(index=False))
    print(json.dumps(spend, indent=2))


if __name__ == "__main__":
    main()
