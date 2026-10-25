"""Check measured spend plus a conservative allowance before starting paid compute.

Wall-clock bounds include image builds and queue time, so they deliberately overestimate
compute. Modal's $40 workspace usage limit remains the external hard limit.
"""
import argparse
import json
from pathlib import Path

# USD / GPU-second, verified at https://modal.com/pricing on 2026-10-24.
RATES = {"H100": 0.001097, "A10G": 0.000306, "A10": 0.000306}
# Additional allowance for CPU / RAM / shutdown overhead, not a measured charge.
OVERHEAD_PER_SECOND = 0.00025


def rate(gpu):
    name, _, count = gpu.partition(":")
    return RATES[name] * int(count or 1) + OVERHEAD_PER_SECOND


def reported_usage(summary):
    # A fresh builder charge may appear in the breakdown before the total catches up.
    breakdown = sum(max(0.0, float(v)) for v in summary.get("metered_cost_breakdown", {}).values())
    return max(float(summary["metered_cost"]), breakdown)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--billing", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--max-minutes", type=float, default=45)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--cap", type=float, default=40)
    a = p.parse_args()
    measured = reported_usage(json.loads(Path(a.billing).read_text()))
    sessions = [json.loads(f.read_text()) for f in Path("logs/modal").glob("session_*.json")]
    bound = sum(s["seconds"] * rate(s["gpu"]) for s in sessions)
    allowance = a.max_minutes * 60 * rate(a.gpu)
    print(f"[budget] measured workspace usage ${measured:.4f}; cumulative wall-time allowance ${bound:.4f}; "
          f"next session allowance ${allowance:.4f}; current cap ${a.cap:g}")
    if not a.report_only and max(measured, bound) + allowance > a.cap:
        raise SystemExit(f"STOP: next session could exceed ${a.cap:g}")


if __name__ == "__main__":
    main()
