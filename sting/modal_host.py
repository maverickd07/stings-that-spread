"""Serve one open-weight model with vLLM on Modal for the length of a command, then shut it down.

  python -m sting.modal_host --model Qwen/Qwen2.5-72B-Instruct --gpu H100:2 -- \
      python -m sting.run snapshots --out runs/q72 --n 8 --models Qwen/Qwen2.5-72B-Instruct

The command sees STING_BASE_URL / STING_API_KEY pointing at the server, so the harness needs no other key.
Differences from the starter's poseidon_compute.modal_serve: tensor parallelism across the GPUs asked for,
one container only (max_containers=1) taking many concurrent requests, and weights cached in a Modal Volume
so a second run of the same model skips the download. Gated models (Llama) need a Modal secret named
"huggingface" with HF_TOKEN.

Rough sizes: 7-8B -> A10G; 32B -> H100; 70-72B -> H100:2.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

MODEL = os.environ.get("STING_MODAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("STING_MODAL_GPU", "A10G")
MAX_LEN = int(os.environ.get("STING_MODAL_MAX_LEN", "8192"))
PORT = 8000
N_GPU = int(GPU.split(":")[1]) if ":" in GPU else 1

app = modal.App("sting-vllm")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.2", "huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)
cache = modal.Volume.from_name("sting-hf-cache", create_if_missing=True)
# Local env vars do not reach the container, so the server's config travels as a secret.
_cfg = {"STING_API_KEY": os.environ.get("STING_API_KEY", ""), "STING_MODAL_MODEL": MODEL,
        "STING_MODAL_MAX_LEN": str(MAX_LEN), "STING_MODAL_NGPU": str(N_GPU),
        "STING_MODAL_GPU": GPU, "STING_MODAL_MAX_SEQS": os.environ.get("STING_MODAL_MAX_SEQS", "16"),
        "STING_MODAL_MAX_BATCHED": os.environ.get("STING_MODAL_MAX_BATCHED", "2048"),
        "STING_MODAL_HF_SECRET": os.environ.get("STING_MODAL_HF_SECRET", "")}
# Modal hydrates dependencies by position: local and remote imports must declare the same objects.
_secrets = [modal.Secret.from_dict(_cfg)]
if os.environ.get("STING_MODAL_HF_SECRET"):
    _secrets.append(modal.Secret.from_name(os.environ["STING_MODAL_HF_SECRET"]))


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, secrets=_secrets, timeout=6 * 3600,
              scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve():
    model = os.environ["STING_MODAL_MODEL"]
    os.environ.setdefault("VLLM_CACHE_ROOT", "/cache/vllm")
    cmd = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", str(PORT), "--served-model-name", model,
           "--tensor-parallel-size", os.environ["STING_MODAL_NGPU"], "--max-model-len", os.environ["STING_MODAL_MAX_LEN"],
           "--gpu-memory-utilization", "0.95", "--max-num-batched-tokens", os.environ.get("STING_MODAL_MAX_BATCHED", "2048"), "--max-num-seqs", os.environ.get("STING_MODAL_MAX_SEQS", "16")]
    if os.environ.get("STING_API_KEY"):
        cmd += ["--api-key", os.environ["STING_API_KEY"]]
    if model == "mistralai/Mistral-Small-24B-Instruct-2501":
        cmd += ["--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral"]
    subprocess.Popen(cmd)


def _wait(url: str, key: str, timeout: int = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    print(f"[modal_host] ready after {time.time() - t0:.0f}s")
                    return
        except Exception:
            pass
        time.sleep(15)
    raise TimeoutError("vLLM server did not come up")


def main() -> None:
    ap = argparse.ArgumentParser(prog="sting.modal_host")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="A10G")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--hf-secret", default="", help="name of a Modal secret holding HF_TOKEN (gated models)")
    ap.add_argument("--max-minutes", type=float, default=45, help="maximum startup + command time")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        sys.exit("give a command after --")
    # Module-level config is read at import, in this process and in the container: re-exec with env set.
    if (os.environ.get("STING_MODAL_MODEL") != args.model or os.environ.get("STING_MODAL_GPU") != args.gpu
            or os.environ.get("STING_MODAL_MAX_LEN") != str(args.max_len)
            or not os.environ.get("STING_API_KEY")):
        env = dict(os.environ, STING_MODAL_MODEL=args.model, STING_MODAL_GPU=args.gpu,
                   STING_MODAL_MAX_LEN=str(args.max_len), STING_MODAL_HF_SECRET=args.hf_secret,
                   STING_API_KEY=os.environ.get("STING_API_KEY") or secrets.token_urlsafe(24))
        os.execvpe(sys.executable, [sys.executable, "-m", "sting.modal_host", *sys.argv[1:]], env)
    key = os.environ["STING_API_KEY"]
    t0 = time.time()
    log_dir = Path(os.environ.get("STING_HOST_LOG_DIR", "logs/modal"))
    log_dir.mkdir(parents=True, exist_ok=True)
    rc = 1
    error = None
    try:
        with modal.enable_output(), app.run():
            url = serve.get_web_url().rstrip("/")
            print(f"[modal_host] {args.model} on {args.gpu} at {url}", flush=True)
            remaining = args.max_minutes * 60 - (time.time() - t0)
            if remaining <= 0:
                raise TimeoutError("Modal build/startup exhausted the command time budget")
            _wait(url, key, timeout=min(1800, remaining))
            rc = subprocess.call(cmd, timeout=max(1, args.max_minutes * 60 - (time.time() - t0)),
                                 env=dict(os.environ, STING_BASE_URL=f"{url}/v1", STING_API_KEY=key,
                                          STING_HOSTING_PROVIDER="modal"))
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        record = {"model": args.model, "gpu": args.gpu, "provider": "modal", "engine": "vllm==0.10.1.1",
                  "started_at": t0, "finished_at": time.time(), "seconds": time.time() - t0,
                  "max_model_len": args.max_len, "max_minutes": args.max_minutes,
                  "gpu_memory_utilization": 0.95, "max_num_batched_tokens": int(os.environ.get("STING_MODAL_MAX_BATCHED", "2048")), "max_num_seqs": int(os.environ.get("STING_MODAL_MAX_SEQS", "16")),
                  "cmd": cmd, "exit_code": rc, "error": error}
        (log_dir / f"session_{int(t0)}.json").write_text(json.dumps(record, indent=2))
        print(f"[modal_host] stopped after {(time.time() - t0) / 60:.1f} min; command exit code {rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
