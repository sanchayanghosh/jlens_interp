"""Compact GPU runtime for single-shot commitment and open-concept analysis."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from experiments.runtime import (
    SYSTEM_PROMPT,
    _token_candidates,
    extract_answer,
    generate_response,
    load_official_lens_from_checkpoint,
    resolve_positions,
)

WORKSPACE_LAYERS = tuple(range(12, 28))
MOTOR_LAYERS = (28, 29, 30)
READOUT_POSITIONS = (
    "user_opinion_letter",
    "assistant_boundary",
    "first_generated_token",
    "pre_answer",
    "generated_answer",
)

CONCEPT_FAMILIES = {
    "objection": (
        "BUT",
        "but",
        "however",
        "although",
        "despite",
        "false",
        "wrong",
        "incorrect",
        "actually",
        "instead",
        "doubt",
        "conflict",
        "disagree",
    ),
    "social_approval": (
        "agree",
        "approval",
        "helpful",
        "please",
        "preference",
        "reassure",
        "satisfy",
        "support",
        "user",
        "validate",
    ),
    "epistemic": (
        "accurate",
        "correct",
        "evidence",
        "fact",
        "honest",
        "knowledge",
        "reason",
        "truth",
    ),
    "evaluation_awareness": (
        "artificial",
        "benchmark",
        "evaluation",
        "fake",
        "fictional",
        "scenario",
        "simulation",
        "test",
    ),
    "control": (
        "basket",
        "bicycle",
        "bottle",
        "camera",
        "chair",
        "clock",
        "cloud",
        "engine",
        "forest",
        "garden",
        "guitar",
        "mountain",
        "orange",
        "pencil",
        "recipe",
        "river",
        "shirt",
        "street",
        "table",
        "window",
    ),
}


def _family_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    families: dict[str, list[int]] = {}
    for family, words in CONCEPT_FAMILIES.items():
        token_ids = {
            token_id
            for word in words
            for token_id in _token_candidates(tokenizer, word)
        }
        if not token_ids:
            raise ValueError(f"No tokenizer candidates for concept family {family}")
        families[family] = sorted(token_ids)
    return families


def _visible_token(token: str) -> str | None:
    normalized = token.strip().lower().lstrip("ġ▁")
    if len(normalized) < 2 or not re.search(r"[a-z]", normalized):
        return None
    return normalized


def _is_surface_token(token: str, visible_text: str) -> bool:
    normalized = _visible_token(token)
    return normalized is None or normalized in visible_text


def _nested_counts(
    counts: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for (split, condition, position, lens), entry in sorted(counts.items()):
        nested.setdefault(split, {}).setdefault(condition, {}).setdefault(position, {})[
            lens
        ] = {
            "slots": entry["slots"],
            "top_counts": {
                str(token_id): count for token_id, count in entry["top_counts"].most_common()
            },
            "hidden_counts": {
                str(token_id): count
                for token_id, count in entry["hidden_counts"].most_common()
            },
        }
    return nested


def _trace_example(
    wrapper: Any,
    lens: Any,
    tokenizer: Any,
    token_ids: Sequence[int],
    positions: dict[str, int],
    family_ids: dict[str, list[int]],
    top_k: int,
    visible_text: str,
    open_split: str | None,
    condition: str,
    counts: dict[tuple[str, str, str, str], dict[str, Any]],
    token_text: dict[int, str],
) -> dict[str, Any]:
    import torch
    from jlens import ActivationRecorder

    valid_positions = {
        name: positions[name]
        for name in READOUT_POSITIONS
        if name in positions and 0 <= positions[name] < len(token_ids)
    }
    ordered_names = list(valid_positions)
    ordered_positions = [valid_positions[name] for name in ordered_names]
    requested_layers = set(WORKSPACE_LAYERS) | set(MOTOR_LAYERS)
    source_layers = [layer for layer in lens.source_layers if layer in requested_layers]
    ids_tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=wrapper.input_device)
    with ActivationRecorder(wrapper.layers, at=source_layers) as recorder, torch.no_grad():
        wrapper.forward(ids_tensor)

    traces: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"jacobian_lens": [], "logit_lens": []} for name in ordered_names
    }
    for layer in source_layers:
        residual = recorder.activations[layer][0, ordered_positions].detach().float()
        transported = lens.transport(residual, layer)
        readouts = {
            "jacobian_lens": wrapper.unembed(transported).float(),
            "logit_lens": wrapper.unembed(residual).float(),
        }
        for lens_name, logits in readouts.items():
            log_probs = logits.log_softmax(dim=-1)
            family_scores = {
                family: torch.logsumexp(log_probs[:, ids], dim=-1)
                for family, ids in family_ids.items()
            }
            top_indices = log_probs.topk(top_k, dim=-1).indices if top_k else None
            for row, position_name in enumerate(ordered_names):
                traces[position_name][lens_name].append(
                    {
                        "layer": layer,
                        "regime": "workspace" if layer in WORKSPACE_LAYERS else "motor",
                        "family_log_probability": {
                            family: round(float(values[row].item()), 6)
                            for family, values in family_scores.items()
                        },
                    }
                )
                if open_split is None or condition not in {"sycophantic", "resistant"}:
                    continue
                key = (open_split, condition, position_name, lens_name)
                entry = counts.setdefault(
                    key,
                    {"slots": 0, "top_counts": Counter(), "hidden_counts": Counter()},
                )
                if layer in WORKSPACE_LAYERS:
                    entry["slots"] += 1
                    for token_id in top_indices[row].tolist():
                        token_id = int(token_id)
                        text = token_text.setdefault(token_id, tokenizer.decode([token_id]))
                        entry["top_counts"][token_id] += 1
                        if not _is_surface_token(text, visible_text):
                            entry["hidden_counts"][token_id] += 1
    return {
        "positions": {
            name: {
                "index": index,
                "token": tokenizer.decode([int(token_ids[index])]),
            }
            for name, index in valid_positions.items()
        },
        "family_traces": traces,
    }


def run_commitment_payload(
    payload: dict[str, Any],
    model_id: str,
    lens_repo: str,
    lens_filename: str,
    cache_dir: str,
    max_new_tokens: int = 256,
    top_k: int = 25,
    system_prompt: str = SYSTEM_PROMPT,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Run compact full-vocabulary readouts on matched single-shot examples."""
    import datetime
    import importlib.metadata
    import time

    import torch
    from huggingface_hub import hf_hub_download
    from jlens import from_hf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    wrapper = from_hf(hf_model, tokenizer, force_bos=False)
    lens_path = hf_hub_download(lens_repo, filename=lens_filename, cache_dir=cache_dir)
    lens, lens_metadata = load_official_lens_from_checkpoint(lens_path, wrapper.n_layers)
    if lens.d_model != wrapper.d_model:
        raise ValueError(
            f"Lens d_model={lens.d_model} is incompatible with model d_model={wrapper.d_model}"
        )
    lens.jacobians = {
        layer: matrix.float().to(hf_model.device) for layer, matrix in lens.jacobians.items()
    }
    family_ids = _family_token_ids(tokenizer)
    counts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    token_text: dict[int, str] = {}
    results: list[dict[str, Any]] = []
    selected_examples = payload["examples"][:max_examples]
    for index, example in enumerate(selected_examples, start=1):
        response, prompt_ids, generated_ids = generate_response(
            hf_model,
            tokenizer,
            example["prompt"],
            max_new_tokens,
            system_prompt,
        )
        answer = extract_answer(response)
        positions = resolve_positions(
            tokenizer,
            prompt_ids,
            generated_ids,
            answer,
            example["user_letter"],
        )
        trace = _trace_example(
            wrapper,
            lens,
            tokenizer,
            [*prompt_ids, *generated_ids],
            positions,
            family_ids,
            top_k,
            (example["prompt"] + "\n" + response).lower(),
            example.get("open_split"),
            example["condition"],
            counts,
            token_text,
        )
        results.append(
            {
                **{key: value for key, value in example.items() if key != "prompt"},
                "response": response,
                "generated_answer": answer,
                "label_stable": answer == example["expected_answer"],
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated_ids),
                **trace,
            }
        )
        if index % 25 == 0:
            print(
                f"Completed {index}/{len(selected_examples)} examples in "
                f"{time.time() - started:.1f}s",
                flush=True,
            )

    return {
        "metadata": {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "model_id": model_id,
            "model_dtype": "bfloat16",
            "decoding": "greedy",
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "system_prompt": system_prompt,
            "lens_repo": lens_repo,
            "lens_filename": lens_filename,
            "lens_metadata": lens_metadata,
            "anthropic_jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "workspace_layers": list(WORKSPACE_LAYERS),
            "motor_layers": list(MOTOR_LAYERS),
            "readout_positions": list(READOUT_POSITIONS),
            "n_examples_completed": len(results),
            "runtime_seconds": round(time.time() - started, 3),
            "gpu": torch.cuda.get_device_name(0),
            "package_versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "jlens", "huggingface_hub")
            },
        },
        "design": payload["design"],
        "concept_families": CONCEPT_FAMILIES,
        "examples": results,
        "concept_counts": _nested_counts(counts),
        "token_text": {str(token_id): text for token_id, text in sorted(token_text.items())},
    }
