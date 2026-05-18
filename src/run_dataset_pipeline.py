#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
PHASE1_RUN = WORKSPACE_ROOT / "pipeline" / "phase1" / "run.py"
PHASE2_RUN = WORKSPACE_ROOT / "pipeline" / "phase2" / "run.py"
PHASE3_RUN = WORKSPACE_ROOT / "pipeline" / "phase3" / "run.py"
DATA_ROOT = WORKSPACE_ROOT / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = "openai/gpt-5.4"
DEFAULT_MAX_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline on every image in a dataset folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Dataset folder or dataset image folder, for example data/Bridge "
            "or data/Bridge/images."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Dataset name under data/. Defaults to the dataset folder name, "
            "for example data/Bridge/images -> Bridge."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root data directory. Default: data.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Phase 1 LiteLLM model string. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Phase 3 random seed for deterministic grammar sampling. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum attempts per pending/failed image. Default: {DEFAULT_MAX_ATTEMPTS}.",
    )
    return parser.parse_args()


def image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_dataset_paths(
    input_dir: Path,
    data_root: Path,
    dataset_name: str | None,
) -> tuple[Path, Path, str]:
    if (input_dir / "images").is_dir():
        dataset_dir = input_dir
        images_dir = input_dir / "images"
    else:
        dataset_dir = input_dir.parent
        images_dir = input_dir

    if dataset_name:
        dataset_dir = data_root / dataset_name
    dataset_name = dataset_name or dataset_dir.name
    return dataset_dir, images_dir, dataset_name


def is_dataset_root(path: Path) -> bool:
    return any(child.is_dir() and (child / "images").is_dir() for child in path.iterdir())


def dataset_jobs(
    input_dir: Path,
    data_root: Path,
    dataset_name: str | None,
) -> list[tuple[Path, Path, str]]:
    if is_dataset_root(input_dir) and not (input_dir / "images").is_dir():
        if dataset_name is not None:
            raise ValueError("--dataset-name cannot be used when --input-dir is a data root")
        return [
            (dataset_dir, dataset_dir / "images", dataset_dir.name)
            for dataset_dir in sorted(input_dir.iterdir())
            if dataset_dir.is_dir() and (dataset_dir / "images").is_dir()
        ]

    return [resolve_dataset_paths(input_dir, data_root, dataset_name)]


def output_path(dataset_dir: Path, phase: str, stem: str) -> Path:
    return dataset_dir / "instructions" / phase / f"{stem}.json"


def run_one_image(
    image_path: Path,
    dataset_dir: Path,
    model: str,
    seed: int,
) -> None:
    stem = image_path.stem
    phase1_output = output_path(dataset_dir, "phase1", stem)
    phase2_output = output_path(dataset_dir, "phase2", stem)
    phase3_output = output_path(dataset_dir, "phase3", stem)

    subprocess.run(
        [
            sys.executable,
            str(PHASE1_RUN),
            "--image",
            str(image_path),
            "--output",
            str(phase1_output),
            "--model",
            model,
        ],
        cwd=WORKSPACE_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PHASE2_RUN),
            "--input",
            str(phase1_output),
            "--output",
            str(phase2_output),
        ],
        cwd=WORKSPACE_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PHASE3_RUN),
            "--input",
            str(phase2_output),
            "--output",
            str(phase3_output),
            "--seed",
            str(seed),
        ],
        cwd=WORKSPACE_ROOT,
        check=True,
    )


def run_one_image_with_retries(
    image_path: Path,
    dataset_dir: Path,
    model: str,
    seed: int,
    max_attempts: int,
) -> None:
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  attempt {attempt}/{max_attempts}")
            run_one_image(
                image_path,
                dataset_dir,
                model,
                seed,
            )
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"  attempt {attempt}/{max_attempts} failed with exit code {exc.returncode}")

    assert last_error is not None
    raise last_error


def is_successful_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return not (isinstance(data, dict) and data.get("status") == "failure")


def image_already_succeeded(image_path: Path, dataset_dir: Path) -> bool:
    stem = image_path.stem
    return all(
        is_successful_json(output_path(dataset_dir, phase, stem))
        for phase in ("phase1", "phase2", "phase3")
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    data_root = args.data_root.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.data_root == DATA_ROOT and is_dataset_root(input_dir):
        data_root = input_dir

    jobs = dataset_jobs(
        input_dir,
        data_root,
        args.dataset_name,
    )

    print("Dataset pipeline run")
    print(f"  input_dir:    {input_dir}")
    print(f"  data_root:    {data_root}")
    print(f"  model:        {args.model}")
    print(f"  seed:         {args.seed}")
    print(f"  max_attempts: {args.max_attempts}")
    print("  resume:       skip images with successful phase1/phase2/phase3 outputs")
    print(f"  datasets:     {len(jobs)}")
    if args.limit is not None:
        print(f"  limit:        {args.limit} per dataset")

    for dataset_index, (dataset_dir, images_dir, dataset_name) in enumerate(jobs, start=1):
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")

        paths = image_paths(images_dir)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            paths = paths[: args.limit]
        if not paths:
            raise ValueError(f"No supported image files found in {images_dir}")

        dataset_dir.mkdir(parents=True, exist_ok=True)
        for phase in ("phase1", "phase2", "phase3"):
            (dataset_dir / "instructions" / phase).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "input_files.txt").write_text(
            "\n".join(str(path.relative_to(images_dir)) for path in paths) + "\n",
            encoding="utf-8",
        )

        print(f"[dataset {dataset_index}/{len(jobs)}] {dataset_name}")
        print(f"  images_dir:  {images_dir}")
        print(f"  phase1-3:    {dataset_dir / 'instructions'}")
        print(f"  image_count: {len(paths)}")
        print(f"  input_list:  {dataset_dir / 'input_files.txt'}")

        for index, image_path in enumerate(paths, start=1):
            if image_already_succeeded(image_path, dataset_dir):
                print(f"[{index}/{len(paths)}] SKIP already succeeded: {image_path}")
                continue
            print(f"[{index}/{len(paths)}] RUN  pending or failed: {image_path}")
            run_one_image_with_retries(
                image_path,
                dataset_dir,
                args.model,
                args.seed,
                args.max_attempts,
            )

    print(f"Complete -> {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
