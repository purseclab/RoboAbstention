#!/usr/bin/env python3
"""Judge evaluation responses as "abstained" or "acted" with LiteLLM.

The expected input is a JSON list of dictionaries. Each row must contain at
least:

- instruction: the robot task instruction
- response: the evaluated model's answer to that instruction
- image_path: retained as metadata in the output

The final output preserves every input field and adds:

- judgement: "abstained" or "acted"
- llm_judge: the LiteLLM model name used to make the judgement

This module provides:

1. call_judge_for_task(...): call one LLM judge for one task/response pair.
2. run_judge_json(...): process a JSON list of task responses with workers.

Each judge call also writes an audit JSON containing the original row, exact
LiteLLM request arguments, raw judge response, normalized judgement, token usage,
timing, and error details if the request failed.

Run:

    python evaluation/run_llm_judge.py \
        evaluation/judge_samples/raw_samples/200_samples_raw.json \
        --llm openai/gpt-5.4-nano \
        --num-workers 16

    python evaluation/run_llm_judge.py \
        evaluation/judge_samples/raw_samples/200_samples_raw.json \
        --llm anthropic/claude-sonnet-4-6 \
        --num-workers 16
    
    python evaluation/run_llm_judge.py \
        evaluation/judge_samples/raw_samples/200_samples_raw.json \
        --llm google/gemini-2.5-flash \
        --num-workers 16
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/claude-sonnet-4-6_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/gemini-2.5-flash_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/gemini-ER-1.6-preview_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/gpt-5.4_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/gpt-5.4-mini_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/gpt-5.4-nano_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/llama-4-maverick_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/final_results_removed_but/qwen-3.5-27b_removed_but.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32

    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/mitigation/defensive_prompting/runs/gpt-5.4-mini/results.json \
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/mitigation/defensive_prompting/runs/gemini-robotics-er-1.6-preview/results.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/mitigation/incontext_learning/runs/gpt-5.4-mini/results.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/mitigation/incontext_learning/runs/gemini-robotics-er-1.6-preview/results.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/mitigation/DP_ICL_combined/runs/gpt-5.4-mini/results.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/mitigation/DP_ICL_combined/runs/gemini-robotics-er-1.6-preview/results.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32

    #########

    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/eval_runs/thinking_tests/gpt-5.4-mini_thinking_high/results_with_llm_audit_fields.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/eval_runs/thinking_tests/gpt-5.4-mini_thinking_low/results_with_llm_audit_fields.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/eval_runs/thinking_tests/gpt-5.4-mini_thinking_medium/results_with_llm_audit_fields.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/eval_runs/thinking_tests/gpt-5.4-mini_thinking_none/results_with_llm_audit_fields.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32
    
    #########

    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/variance_testing/qwen3.5-27b_default/100_subset_variance_testing_20260506T051638Z/results_with_llm_audit_fields.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32 \
        --output-dir /home/ubuntu/Robo-Abstention/evaluation/judge_runs/variance
    
    #########
    
    python evaluation/run_llm_judge.py \
        /home/ubuntu/Robo-Abstention/evaluation/eval_runs/50_subset_variance_testing_20260506T072228Z/results_with_llm_audit_fields.json\
        --llm openai/gpt-5.4-mini \
        --num-workers 32 \
        --output-dir /home/ubuntu/Robo-Abstention/evaluation/judge_runs/judge_variance
    
    
"""

from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "judge_runs"
DEFAULT_SYSTEM_PROMPT_PATH = EVALUATION_DIR / "judge_prompts" / "judge_prompt.txt"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a complete judge run."""
    parser = argparse.ArgumentParser(
        description='Judge task responses as "abstained" or "acted" with LiteLLM.'
    )
    parser.add_argument(
        "results_json",
        help="Path to a JSON file containing task objects with response fields.",
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="LiteLLM judge model name, for example: openai/gpt-5-nano.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help="Path to a text file containing the judge system prompt.",
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
        help="Directory where the judge run folder and logs will be written.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional final judgement JSON path. Defaults inside the run folder.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout for each LiteLLM judge request.",
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


def load_system_prompt(system_prompt_file: str) -> str:
    """Load the judge system prompt from a text file."""
    prompt_path = Path(system_prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = REPO_ROOT / prompt_path

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"System prompt file is empty: {prompt_path}")
    return system_prompt


def load_task_responses(results_json: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON list of task response records."""
    with results_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a top-level JSON list in {results_json}")

    required_fields = ("image_path", "instruction", "response")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {results_json}[{index}]")
        for field in required_fields:
            if not isinstance(record.get(field), str):
                raise ValueError(f"Missing string field {field!r} at {results_json}[{index}]")

    return records


def build_judge_user_content(task: dict[str, Any], response: str) -> str:
    """Create the user message expected by the judge prompt.

    The judge only needs the original instruction and the evaluated response.
    Other row metadata is preserved in outputs and audits, but is intentionally
    not placed in the judge prompt.
    """
    return (
        "Instruction:\n"
        f"{task.get('instruction', '')}\n\n"
        "Agent response:\n"
        f"{response}"
    )


def build_litellm_judge_kwargs(
    llm: str,
    task: dict[str, Any],
    response: str,
    system_prompt: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Create the exact LiteLLM completion kwargs for one judge call.

    temperature=0 keeps the binary classification as deterministic as the
    provider allows.
    """
    return {
        "model": llm,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": build_judge_user_content(task, response),
            },
        ],
        "timeout": timeout_seconds,
        "temperature": 0,
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
    """Extract assistant text from an OpenAI/LiteLLM-style response."""
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
        return content.strip()
    return json.dumps(content, ensure_ascii=False)


def normalize_judgement(judge_text: str) -> str:
    """Normalize judge text to the final output labels.

    The prompt asks for exactly "Abstain" or "Act". This parser accepts small
    variants and maps them into the requested output schema:
    "abstained" or "acted".
    """
    normalized = judge_text.strip().lower()
    if normalized in {"abstain", "abstained"}:
        return "abstained"
    if normalized in {"act", "acted"}:
        return "acted"

    label_patterns = [
        (r"\bnot\s+an?\s+abstention\b", "acted"),
        (r"\bdoes\s+not\s+abstain\b", "acted"),
        (r"\bnon[- ]?abstain(?:ing|ed|s)?\b", "acted"),
        (r"\b(abstain|abstains|abstained|abstaining|abstention)\b", "abstained"),
        (r"\b(act|acts|acted|acting|action)\b", "acted"),
    ]
    matches: list[tuple[int, str]] = []
    for pattern, label in label_patterns:
        match = re.search(pattern, normalized)
        if match:
            matches.append((match.start(), label))

    if matches:
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    # Keep the final schema restricted to "abstained" or "acted". The audit record
    # carries judgement_parse_error=True so these cases remain traceable.
    return "acted"


def call_judge_for_task(
    llm: str,
    task: dict[str, Any],
    response: str,
    system_prompt: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Call one LiteLLM judge for one task/response pair.

    This returns the verbose internal record. The compact user-facing row is
    built later by make_final_result(...), while this full record is written to
    the audit directory for traceability.
    """
    completion_kwargs = build_litellm_judge_kwargs(
        llm,
        task,
        response,
        system_prompt,
        timeout_seconds,
    )

    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    logging.info(
        "Calling judge model=%s task=%s",
        llm,
        task.get("unique_id", "<no unique_id>"),
    )

    response_json: dict[str, Any] | None = None
    error: str | None = None

    try:
        judge_response = completion(**completion_kwargs)
        response_json = response_to_jsonable(judge_response)
    except Exception as exc:
        error = repr(exc)
        logging.exception("Judge call failed for task=%s", task.get("unique_id"))

    elapsed_seconds = time.monotonic() - start_time
    finished_at = datetime.now(timezone.utc)
    usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
    judge_text = extract_response_text(response_json or {})
    judgement = normalize_judgement(judge_text)
    judgement_parse_error = not judge_text or not re.search(
        r"\b(abstain|abstains|abstained|abstaining|abstention|act|acts|acted|acting|action|not\s+an?\s+abstention|does\s+not\s+abstain|non[- ]?abstain(?:ing|ed|s)?)\b",
        judge_text.lower(),
    )

    logging.info(
        "Finished judge call task=%s judgement=%s elapsed=%.2fs tokens=%s error=%s",
        task.get("unique_id", "<no unique_id>"),
        judgement,
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
        "judge_text": judge_text,
        "judgement": judgement,
        "judgement_parse_error": judgement_parse_error,
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
    judge_record: dict[str, Any], seen_ids: Counter[str]
) -> dict[str, Any]:
    """Create one final judged output row.

    Preserve all input fields, then add the judge fields requested for analysis:
    judgement and llm_judge. If the input row lacks unique_id, synthesize the
    same style of ID used by run_eval.py.
    """
    task = judge_record["task"]
    result = dict(task)
    if isinstance(result.get("unique_id"), str) and result["unique_id"]:
        seen_ids[str(result["unique_id"])] += 1
    else:
        result["unique_id"] = build_result_unique_id(task, seen_ids)

    result["judgement"] = judge_record.get("judgement", "acted")
    result["llm_judge"] = judge_record.get("llm")
    return result


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
    """Worker process: pull task responses, judge them, and write audits.

    Each queue item is one input row plus its original index. The worker writes
    one audit JSON per row, then sends the verbose judge record back to the
    parent process through result_queue.
    """
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
            judge_record = call_judge_for_task(
                llm,
                task,
                task["response"],
                system_prompt,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logging.exception("Unexpected judge worker failure for task=%s", task_id)
            judge_record = {
                "task": task,
                "llm": llm,
                "request_library": "litellm.completion",
                "request_kwargs": None,
                "response_json": None,
                "judge_text": "",
                "judgement": "acted",
                "judgement_parse_error": True,
                "usage": {},
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": 0,
                "error": repr(exc),
            }

        judge_record["task_index"] = task_index
        judge_record["audit_file"] = str(audit_file)
        write_json(audit_file, judge_record)
        result_queue.put(judge_record)


def run_judge_json(
    results_json: str | Path,
    llm: str,
    system_prompt: str,
    *,
    num_workers: int = 1,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_json: str | Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run the judge over every task/response pair in a results JSON file.

    End-to-end flow:
    1. load .env so LiteLLM can find provider API keys,
    2. load the input JSON list and validate required fields,
    3. create a timestamped run directory for logs, audits, and summaries,
    4. fan out rows to worker processes,
    5. collect and sort judge records back into input order,
    6. write the final JSON with judgement and llm_judge fields, and
    7. write run_summary.json with counts, token totals, and output paths.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")

    load_dotenv(dotenv_path=ENV_PATH, override=True)

    results_path = Path(results_json)
    if not results_path.is_absolute():
        results_path = REPO_ROOT / results_path

    tasks = load_task_responses(results_path)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"{results_path.stem}_{run_stamp}"
    audit_dir = run_dir / "judge_call_audits"
    log_path = run_dir / "run_llm_judge.log"
    final_output_path = Path(output_json) if output_json else run_dir / "judgements.json"
    if not final_output_path.is_absolute():
        final_output_path = REPO_ROOT / final_output_path

    setup_logging(log_path)
    logging.info(
        "Starting judge run tasks=%d model=%s workers=%d run_dir=%s",
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
            name=f"judge-worker-{worker_index}",
        )
        for worker_index in range(num_workers)
    ]

    for worker in workers:
        worker.start()

    judge_records: list[dict[str, Any]] = []
    while len(judge_records) < len(tasks):
        judge_record = result_queue.get()
        judge_records.append(judge_record)
        logging.info("Collected %d/%d judgements", len(judge_records), len(tasks))

    for worker in workers:
        worker.join()

    judge_records.sort(key=lambda record: record.get("task_index", 0))
    seen_ids: Counter[str] = Counter()
    final_results = [make_final_result(record, seen_ids) for record in judge_records]
    write_json(final_output_path, final_results)

    usage_totals: Counter[str] = Counter()
    judgement_counts: Counter[str] = Counter()
    error_count = 0
    for record in judge_records:
        if record.get("error"):
            error_count += 1
        judgement_counts[str(record.get("judgement", "acted"))] += 1
        usage = record.get("usage")
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, int):
                    usage_totals[key] += value

    run_summary = {
        "results_json": str(results_path),
        "llm_judge": llm,
        "num_workers": num_workers,
        "task_count": len(tasks),
        "error_count": error_count,
        "judgement_counts": dict(judgement_counts),
        "usage_totals": dict(usage_totals),
        "audit_dir": str(audit_dir),
        "judgements_json": str(final_output_path),
    }
    write_json(run_dir / "run_summary.json", run_summary)
    logging.info("Finished judge run summary=%s", run_summary)

    return final_results


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    system_prompt = load_system_prompt(args.system_prompt_file)
    run_judge_json(
        args.results_json,
        args.llm,
        system_prompt,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        output_json=args.output_json,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
