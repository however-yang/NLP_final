from __future__ import annotations

import time
from pathlib import Path

from text_rich_mllm.models.generation_utils import run_generation
from text_rich_mllm.prompts import PromptBuilder


def generate_predictions(
    *,
    samples,
    model,
    processor,
    prompt_style: str,
    generation_config: dict,
    output_path: str | Path | None = None,
    existing_predictions: dict[str, str] | None = None,
    limit: int | None = None,
    continue_on_error: bool = False,
):
    from tqdm.auto import tqdm

    from text_rich_mllm.utils import write_jsonl

    builder = PromptBuilder(style=prompt_style)
    prediction_map = dict(existing_predictions or {})
    output_records = [{"sample_id": key, "prediction": value} for key, value in prediction_map.items()]

    iterable = list(samples)
    if limit is not None:
        iterable = iterable[:limit]

    to_process = [sample for sample in iterable if sample.sample_id not in prediction_map]
    resumed = len(iterable) - len(to_process)
    total_slots = len(iterable)

    print(
        "\n[inference] "
        f"prompt_style={prompt_style!r} "
        f"total_samples={total_slots}"
        + (f" limit={limit}" if limit is not None else "")
        + f" resume_skipped={resumed} "
        f"to_generate={len(to_process)} "
        f"output={output_path!s}",
        flush=True,
    )

    if not to_process:
        print("[inference] nothing to generate (all sample_ids already in predictions).", flush=True)
        return prediction_map

    t0 = time.perf_counter()
    pbar = tqdm(
        to_process,
        desc=f"infer[{prompt_style}]",
        unit="sample",
        total=len(to_process),
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for i, sample in enumerate(pbar, start=1):
        try:
            prediction = run_generation(
                model,
                processor,
                sample.image_path,
                builder.build(sample),
                generation_config,
            )
        except Exception:
            if not continue_on_error:
                raise
            prediction = ""
        prediction_map[sample.sample_id] = prediction
        output_records.append({"sample_id": sample.sample_id, "prediction": prediction})
        if output_path:
            write_jsonl(output_records, output_path)

        elapsed = time.perf_counter() - t0
        if elapsed > 0:
            pbar.set_postfix(rate=f"{i / elapsed:.2f}/s", refresh=False)

    elapsed_total = time.perf_counter() - t0
    rate_mean = len(to_process) / elapsed_total if elapsed_total > 0 else 0.0
    print(
        f"\n[inference] finished in {elapsed_total:.1f}s "
        f"({rate_mean:.3f} samples/s on generated subset; "
        f"{elapsed_total / max(len(to_process), 1):.2f}s/sample wall avg).",
        flush=True,
    )

    return prediction_map
