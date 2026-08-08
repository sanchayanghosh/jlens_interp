"""Modal entrypoints for the Llama-8B Jacobian Lens sycophancy experiment.

Examples:
    modal run experiments/modal_app.py::inspect_assets
    modal run experiments/modal_app.py::smoke --model unsloth/Meta-Llama-3.1-8B-Instruct
    modal run experiments/modal_app.py::run --model unsloth/Meta-Llama-3.1-8B-Instruct
    modal run experiments/modal_app.py::run_expanded --model unsloth/Meta-Llama-3.1-8B-Instruct

Override the default H100 with ``JLENS_MODAL_GPU=A100-80GB`` when desired.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/jlens_interp"
GPU = os.environ.get("JLENS_MODAL_GPU", "H100")
DEFAULT_LENS_REPO = "Kameshr/jspace-lens-Llama8b"
DEFAULT_LENS_FILENAME = "jlens.pt"
DEFAULT_CASES = REPO_ROOT / "experiments" / "selected_cases.json"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results" / "raw_results.json"
DEFAULT_EXPANDED_CASES = REPO_ROOT / "experiments" / "expanded_cases.json"
DEFAULT_EXPANDED_RESULTS = (
    REPO_ROOT / "experiments" / "results" / "expanded_summary.json"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.8,<3",
        "transformers>=5.5,<6",
        "accelerate>=1.0,<2",
        "huggingface_hub>=0.28,<2",
        "safetensors>=0.4,<1",
        "numpy>=1.26,<3",
        "git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e",
    )
    .env({"PYTHONPATH": REMOTE_ROOT, "HF_HOME": "/cache/huggingface"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git/**",
            ".venv/**",
            "**/__pycache__/**",
            "output/**",
            "experiments/results/*.json",
        ],
    )
)

app = modal.App("jlens-sycophancy")
hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
VOLUMES = {"/cache": hf_cache}


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def inspect_assets(
    lens_repo: str = DEFAULT_LENS_REPO,
    lens_filename: str = DEFAULT_LENS_FILENAME,
) -> dict[str, Any]:
    """Download the public lens and report only metadata, keys, and matrix shapes."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(lens_repo, filename=lens_filename, cache_dir="/cache/huggingface")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    matrices = payload.get("J", payload.get("avg_jacobians", {}))
    first = matrices[sorted(matrices)[0]] if matrices else None
    result = {
        "keys": sorted(payload),
        "n_matrices": len(matrices),
        "first_shape": list(first.shape) if first is not None else None,
        "first_dtype": str(first.dtype) if first is not None else None,
        "model_id": payload.get("model_id"),
        "d_model": payload.get("d_model"),
        "n_prompts": payload.get("n_prompts"),
        "n_proj": payload.get("n_proj"),
    }
    hf_cache.commit()
    print(json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_remote(
    payload: dict[str, Any],
    model: str,
    lens_repo: str,
    lens_filename: str,
    max_new_tokens: int,
    top_k: int,
    max_prompts: int | None,
) -> dict[str, Any]:
    from experiments.runtime import run_payload

    result = run_payload(
        payload,
        model,
        lens_repo,
        lens_filename,
        cache_dir="/cache/huggingface",
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        max_prompts=max_prompts,
    )
    hf_cache.commit()
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=7200)
def run_expanded_remote(
    payload: dict[str, Any],
    model: str,
    lens_repo: str,
    lens_filename: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run the large grid and return aggregates rather than hundreds of MB of traces."""
    from experiments.analysis import analyze_expanded_results
    from experiments.runtime import run_payload

    result = run_payload(
        payload,
        model,
        lens_repo,
        lens_filename,
        cache_dir="/cache/huggingface",
        max_new_tokens=max_new_tokens,
        top_k=0,
    )
    summary = analyze_expanded_results(result)
    hf_cache.commit()
    return summary


def _execute(
    cases_path: str,
    output_path: str,
    model: str,
    lens_repo: str,
    lens_filename: str,
    max_new_tokens: int,
    top_k: int,
    max_prompts: int | None,
) -> None:
    payload = json.loads(Path(cases_path).expanduser().read_text())
    result = run_remote.remote(
        payload,
        model,
        lens_repo,
        lens_filename,
        max_new_tokens,
        top_k,
        max_prompts,
    )
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metadata"], indent=2))
    print(f"Wrote {destination}")


@app.local_entrypoint()
def smoke(
    model: str,
    cases_path: str = str(DEFAULT_CASES),
    output_path: str = str(REPO_ROOT / "experiments" / "results" / "smoke_results.json"),
    lens_repo: str = DEFAULT_LENS_REPO,
    lens_filename: str = DEFAULT_LENS_FILENAME,
    max_new_tokens: int = 64,
    top_k: int = 5,
) -> None:
    """Run one prompt through generation plus both lenses."""
    _execute(
        cases_path,
        output_path,
        model,
        lens_repo,
        lens_filename,
        max_new_tokens,
        top_k,
        1,
    )


@app.local_entrypoint()
def run(
    model: str,
    cases_path: str = str(DEFAULT_CASES),
    output_path: str = str(DEFAULT_RESULTS),
    lens_repo: str = DEFAULT_LENS_REPO,
    lens_filename: str = DEFAULT_LENS_FILENAME,
    max_new_tokens: int = 128,
    top_k: int = 8,
) -> None:
    """Run all 24 prompts and save exact layer/position traces locally."""
    _execute(
        cases_path,
        output_path,
        model,
        lens_repo,
        lens_filename,
        max_new_tokens,
        top_k,
        None,
    )


@app.local_entrypoint()
def run_expanded(
    model: str,
    cases_path: str = str(DEFAULT_EXPANDED_CASES),
    output_path: str = str(DEFAULT_EXPANDED_RESULTS),
    lens_repo: str = DEFAULT_LENS_REPO,
    lens_filename: str = DEFAULT_LENS_FILENAME,
    max_new_tokens: int = 256,
) -> None:
    """Run 936 expanded prompts and save item-level plus layerwise aggregates."""
    payload = json.loads(Path(cases_path).expanduser().read_text())
    summary = run_expanded_remote.remote(
        payload,
        model,
        lens_repo,
        lens_filename,
        max_new_tokens,
    )
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metadata"], indent=2))
    print(json.dumps(summary["regenerated_behavior_counts"], indent=2))
    print(f"Wrote {destination}")
