#!/usr/bin/env python3
"""Run multimodal evaluation tasks through LiteLLM.

This module provides:

1. call_llm_for_task(...): call one vision-capable LLM with one task and image.
2. run_task_json(...): process a JSON list of tasks with a multiprocessing queue.

Each LLM call writes a complete audit JSON containing the original task, the
exact LiteLLM request arguments, the raw response, token usage, timing, and
error details if the request failed.

Run:

    python evaluation/run_eval.py \
        /home/ubuntu/Robo-Abstention/evaluation/subset_tasks_removed_but.json \
        --llm openrouter/qwen/qwen3.5-9b \
        --system-prompt-file evaluation/system_prompts/default.txt \
        --num-workers 16
    
    python evaluation/run_eval.py \
        /home/ubuntu/Robo-Abstention/evaluation/subset_tasks_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --system-prompt-file evaluation/system_prompts/default.txt \
        --num-workers 16
    
    python evaluation/run_eval.py \
        /home/ubuntu/Robo-Abstention/evaluation/variance_testing/50_subset_variance_testing.json \
        --llm openai/gpt-5.4-mini \
        --system-prompt-file evaluation/system_prompts/default.txt \
        --num-workers 32 \
        --output-dir /home/ubuntu/Robo-Abstention/evaluation/judge_runs/judge_variance
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import multiprocessing as mp
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any

from dotenv import load_dotenv
from litellm import completion


REPO_ROOT = Path("/home/ubuntu/Robo-Abstention")
EVALUATION_DIR = REPO_ROOT / "evaluation"
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "eval_runs"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a complete evaluation run."""
    parser = argparse.ArgumentParser(
        description="Run a task JSON through a LiteLLM multimodal LLM."
    )
    parser.add_argument(
        "task_json",
        help="Path to a JSON file containing a list of task objects.",
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="LiteLLM model name, for example: openai/gpt-5-nano.",
    )
    parser.add_argument(
        "--system-prompt-file",
        required=True,
        help="Path to a text file containing the system prompt for every task.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the run folder and logs will be written.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional final results JSON path. Defaults inside the run folder.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout for each LiteLLM request.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path | None = None) -> None:
    """Configure process-safe console/file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def load_tasks(task_json: Path) -> list[dict[str, Any]]:
    """Load and validate a task JSON list."""
    with task_json.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError(f"Expected a top-level JSON list in {task_json}")

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Expected object at {task_json}[{index}]")
        if not isinstance(task.get("image_path"), str) or not task["image_path"]:
            raise ValueError(f"Missing image_path at {task_json}[{index}]")
        if not isinstance(task.get("instruction"), str) or not task["instruction"]:
            raise ValueError(f"Missing instruction at {task_json}[{index}]")

    return tasks


def load_system_prompt(system_prompt_file: str) -> str:
    """Load the system prompt from a text file."""
    prompt_path = Path(system_prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = REPO_ROOT / prompt_path

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"System prompt file is empty: {prompt_path}")
    return system_prompt


def encode_image_as_data_url(image_path: Path) -> str:
    """Read a local image and return a data URL suitable for vision LLMs."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_litellm_kwargs(
    task: dict[str, Any],
    image_path: Path,
    llm: str,
    system_prompt: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Create the exact LiteLLM completion kwargs for one task."""
    image_data_url = encode_image_as_data_url(image_path)

    return {
        "model": llm,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": task["instruction"],
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            },
        ],
        "timeout": timeout_seconds,
    }


def response_to_jsonable(response: Any) -> dict[str, Any]:
    """Convert a LiteLLM response object into JSON-serializable data."""
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    return json.loads(json.dumps(response, default=str))


def extract_response_text(response_json: dict[str, Any]) -> str:
    """Extract the assistant text from an OpenAI/LiteLLM-style response."""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def call_llm_for_task(
    task: dict[str, Any],
    image: str | Path,
    llm: str,
    system_prompt: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Call one LiteLLM-supported model with one task, image, and system prompt.

    The user message contains the task instruction as text and the image in the
    image_url field. The returned dictionary includes the exact request payload,
    raw response JSON, extracted response text, token usage, timing, and errors.
    """
    image_path = Path(image)
    completion_kwargs = build_litellm_kwargs(
        task,
        image_path,
        llm,
        system_prompt,
        timeout_seconds,
    )

    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    logging.info(
        "Calling LLM model=%s task=%s image=%s",
        llm,
        task.get("unique_id", "<no unique_id>"),
        image_path,
    )

    response_json: dict[str, Any] | None = None
    error: str | None = None

    try:
        response = completion(**completion_kwargs)
        response_json = response_to_jsonable(response)
    except Exception as exc:
        error = repr(exc)
        logging.exception("LLM call failed for task=%s", task.get("unique_id"))

    elapsed_seconds = time.monotonic() - start_time
    finished_at = datetime.now(timezone.utc)
    usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
    response_text = extract_response_text(response_json or {})

    logging.info(
        "Finished LLM call task=%s elapsed=%.2fs tokens=%s error=%s",
        task.get("unique_id", "<no unique_id>"),
        elapsed_seconds,
        usage,
        error,
    )

    return {
        "task": task,
        "llm": llm,
        "request_library": "litellm.completion",
        "request_kwargs": completion_kwargs,
        "response_json": response_json,
        "response_text": response_text,
        "usage": usage,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }


def slugify(value: str, *, lowercase: bool = True) -> str:
    """Make a value safe for use inside the final unique_id."""
    value = value.strip()
    if lowercase:
        value = value.lower()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_") or "empty"


def build_result_unique_id(task: dict[str, Any], seen_ids: Counter[str]) -> str:
    """Build benchmark__image__type__instruction IDs and suffix duplicates."""
    benchmark = str(task.get("benchmark", "unknown_benchmark"))
    image_stem = Path(str(task.get("image_path", "unknown_image"))).stem
    task_type = str(task.get("type", "unknown_type"))
    instruction = str(task.get("instruction", "unknown_instruction"))

    base_id = "__".join(
        [
            slugify(benchmark, lowercase=False),
            slugify(image_stem),
            slugify(task_type),
            slugify(instruction),
        ]
    )
    seen_ids[base_id] += 1
    if seen_ids[base_id] == 1:
        return base_id
    return f"{base_id}__{seen_ids[base_id]}"


def make_final_result(
    call_record: dict[str, Any], seen_ids: Counter[str]
) -> dict[str, Any]:
    """Create the requested compact final output record."""
    task = call_record["task"]
    return {
        "unique_id": build_result_unique_id(task, seen_ids),
        "benchmark": task.get("benchmark"),
        "image_path": task.get("image_path"),
        "instruction": task.get("instruction"),
        "type": task.get("type"),
        "response": call_record.get("response_text", ""),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    *,
    llm: str,
    system_prompt: str,
    audit_dir: str,
    timeout_seconds: int,
) -> None:
    """Worker process: pull tasks, call the LLM, write audits, return results."""
    setup_logging()
    audit_path = Path(audit_dir)

    while True:
        try:
            queue_item = task_queue.get(timeout=1)
        except Empty:
            continue

        if queue_item is None:
            return

        task_index, task = queue_item
        task_id = task.get("unique_id") or slugify(task["instruction"])
        safe_task_id = slugify(str(task_id), lowercase=False)
        audit_file = audit_path / f"{task_index:06d}_{safe_task_id}.json"

        try:
            call_record = call_llm_for_task(
                task,
                task["image_path"],
                llm,
                system_prompt,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logging.exception("Unexpected worker failure for task=%s", task_id)
            call_record = {
                "task": task,
                "llm": llm,
                "request_library": "litellm.completion",
                "request_kwargs": None,
                "response_json": None,
                "response_text": "",
                "usage": {},
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": 0,
                "error": repr(exc),
            }

        call_record["task_index"] = task_index
        call_record["audit_file"] = str(audit_file)
        write_json(audit_file, call_record)
        result_queue.put(call_record)


def run_task_json(
    task_json: str | Path,
    llm: str,
    system_prompt: str,
    *,
    num_workers: int = 1,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_json: str | Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run every task in a JSON file with a multiprocessing queue.

    Returns the compact final result list and writes it to output_json.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")

    load_dotenv(dotenv_path=ENV_PATH, override=True)

    task_path = Path(task_json)
    if not task_path.is_absolute():
        task_path = REPO_ROOT / task_path

    tasks = load_tasks(task_path)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"{task_path.stem}_{run_stamp}"
    audit_dir = run_dir / "llm_call_audits"
    log_path = run_dir / "run_eval.log"
    final_output_path = Path(output_json) if output_json else run_dir / "results.json"
    if not final_output_path.is_absolute():
        final_output_path = REPO_ROOT / final_output_path

    setup_logging(log_path)
    logging.info(
        "Starting run tasks=%d model=%s workers=%d run_dir=%s",
        len(tasks),
        llm,
        num_workers,
        run_dir,
    )

    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    for task_index, task in enumerate(tasks):
        task_queue.put((task_index, task))
    for _ in range(num_workers):
        task_queue.put(None)

    workers = [
        mp.Process(
            target=worker_loop,
            kwargs={
                "task_queue": task_queue,
                "result_queue": result_queue,
                "llm": llm,
                "system_prompt": system_prompt,
                "audit_dir": str(audit_dir),
                "timeout_seconds": timeout_seconds,
            },
            name=f"eval-worker-{worker_index}",
        )
        for worker_index in range(num_workers)
    ]

    for worker in workers:
        worker.start()

    call_records: list[dict[str, Any]] = []
    while len(call_records) < len(tasks):
        call_record = result_queue.get()
        call_records.append(call_record)
        logging.info("Collected %d/%d results", len(call_records), len(tasks))

    for worker in workers:
        worker.join()

    call_records.sort(key=lambda record: record.get("task_index", 0))
    seen_ids: Counter[str] = Counter()
    final_results = [make_final_result(record, seen_ids) for record in call_records]
    write_json(final_output_path, final_results)

    usage_totals: Counter[str] = Counter()
    error_count = 0
    for record in call_records:
        if record.get("error"):
            error_count += 1
        usage = record.get("usage")
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, int):
                    usage_totals[key] += value

    run_summary = {
        "task_json": str(task_path),
        "llm": llm,
        "num_workers": num_workers,
        "task_count": len(tasks),
        "error_count": error_count,
        "usage_totals": dict(usage_totals),
        "audit_dir": str(audit_dir),
        "results_json": str(final_output_path),
    }
    write_json(run_dir / "run_summary.json", run_summary)
    logging.info("Finished run summary=%s", run_summary)

    return final_results


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    system_prompt = load_system_prompt(args.system_prompt_file)
    run_task_json(
        args.task_json,
        args.llm,
        system_prompt,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        output_json=args.output_json,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
