#!/usr/bin/env python3

import json
import os
import shlex
import subprocess
import sys
import time
import argparse

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Experiment:
    name: str
    opacity_entropy: bool
    entropy_start: str
    prune_and_recovery: bool


EXPERIMENTS = [
    Experiment(
        name="oe_early",
        opacity_entropy=True,
        entropy_start="densify_from_iter",
        prune_and_recovery=False,
    ),
    Experiment(
        name="oe_late",
        opacity_entropy=True,
        entropy_start="densify_until_iter",
        prune_and_recovery=False,
    ),
    Experiment(
        name="oe_early_pr",
        opacity_entropy=True,
        entropy_start="densify_from_iter",
        prune_and_recovery=True,
    ),
    Experiment(
        name="oe_late_pr",
        opacity_entropy=True,
        entropy_start="densify_until_iter",
        prune_and_recovery=True,
    ),
]

# ============================================================
# LOGGING
# ============================================================


def log(message, logfile):
    print(message, flush=True)
    logfile.write(message + "\n")
    logfile.flush()


def run_command(label, command, logfile):
    log("\n" + "=" * 80, logfile)
    log(label, logfile)
    log("$ " + shlex.join(command), logfile)
    log("=" * 80, logfile)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    start = time.monotonic()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="", flush=True)
        logfile.write(line)
        logfile.flush()

    return_code = process.wait()
    elapsed = time.monotonic() - start

    log(
        f"\n[{label}] Exit code: {return_code} | " f"Elapsed: {elapsed:.2f}s",
        logfile,
    )

    return return_code, elapsed


EXPERIMENTS = [
    Experiment(
        name="oe_early",
        opacity_entropy=True,
        entropy_start="densify_from_iter",
        prune_and_recovery=False,
    ),
    Experiment(
        name="oe_late",
        opacity_entropy=True,
        entropy_start="densify_until_iter",
        prune_and_recovery=False,
    ),
    Experiment(
        name="oe_early_pr",
        opacity_entropy=True,
        entropy_start="densify_from_iter",
        prune_and_recovery=True,
    ),
    Experiment(
        name="oe_late_pr",
        opacity_entropy=True,
        entropy_start="densify_until_iter",
        prune_and_recovery=True,
    ),
]
# ============================================================
# EXPERIMENT
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="CoCO-Train Automated Ablation Runner")
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        required=True,
        help="Path to COLMAP scene directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Base output directory for the ablation run",
    )
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[7_000, 30_000])
    parser.add_argument("--prune_iteration", type=int, default=20_000)
    parser.add_argument("--prune_percent", type=float, default=0.20)
    parser.add_argument("--lambda_opacity", type=float, default=0.005)
    parser.add_argument(
        "--tight_aabb",
        action="store_true",
        help="Enable opacity-aware tight AABB in CoCO-DGR",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[e.name for e in EXPERIMENTS],
        help="Run only selected experiment variants",
    )
    return parser.parse_args()


def run_experiment(exp, args, run_root, logfile):

    model_path = run_root / exp.name
    model_path.mkdir(parents=True, exist_ok=True)

    log("\n\n" + "#" * 80, logfile)
    log(f"EXPERIMENT: {exp.name}", logfile)
    log("#" * 80, logfile)

    config_path = model_path / "experiment_config.json"

    with config_path.open("w") as f:
        json.dump(asdict(exp), f, indent=4)

    # --------------------------------------------------------
    # 1. TRAINING
    # --------------------------------------------------------

    train_cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(args.source.resolve()),
        "-m",
        str(model_path),
        "--eval",
        "--optimizer_type",
        "default",
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        *map(str, args.checkpoints),
        "--save_iterations",
        *map(str, args.checkpoints),
        "--opacity_entropy_start",
        exp.entropy_start,
        f"--lambda_opacity={args.lambda_opacity}",
        "--prune_iteration",
        str(args.prune_iteration),
        "--prune_percent",
        str(args.prune_percent),
    ]

    if args.tight_aabb:
        train_cmd.append("--tight_aabb")
    if exp.opacity_entropy:
        train_cmd.append("--opacity_entropy")
    if exp.prune_and_recovery:
        train_cmd.append("--prune_and_recovery")
    if exp.opacity_entropy:
        train_cmd.append("--opacity_entropy")

    if exp.prune_and_recovery:
        train_cmd.append("--prune_and_recovery")

    code, train_time = run_command(
        f"TRAINING: {exp.name}",
        train_cmd,
        logfile,
    )

    if code != 0:
        log(
            f"[FAILED] Training failed for {exp.name}. "
            "Skipping rendering and metrics.",
            logfile,
        )
        return

    # --------------------------------------------------------
    # 2. CHECKPOINT VALIDATION
    # --------------------------------------------------------

    missing = []

    for iteration in args.checkpoints:
        ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

        if not ply.exists():
            missing.append(iteration)

    if missing:
        log(
            f"[FAILED] Missing checkpoints: {missing}",
            logfile,
        )
        return

    # --------------------------------------------------------
    # 3. RENDERING
    # --------------------------------------------------------

    for iteration in args.checkpoints:

        render_cmd = [
            sys.executable,
            "render.py",
            "-m",
            str(model_path),
            "--iteration",
            str(iteration),
            "--skip_train",
        ]

        code, _ = run_command(
            f"RENDER {iteration}: {exp.name}",
            render_cmd,
            logfile,
        )

        if code != 0:
            log(
                f"[FAILED] Render failed at iteration {iteration}.",
                logfile,
            )
            return

    # --------------------------------------------------------
    # 4. METRICS
    # --------------------------------------------------------

    metrics_cmd = [
        sys.executable,
        "metrics.py",
        "-m",
        str(model_path),
    ]

    code, _ = run_command(
        f"METRICS: {exp.name}",
        metrics_cmd,
        logfile,
    )

    if code != 0:
        log("[FAILED] Metrics calculation failed.", logfile)
        return

    # --------------------------------------------------------
    # 5. COLLECT RESULTS
    # --------------------------------------------------------

    results_path = model_path / "results.json"
    stats_path = model_path / "training_stats.txt"

    if results_path.exists():
        with results_path.open() as f:
            results = json.load(f)

        log("\nRESULTS:", logfile)

        for method, metrics in results.items():
            log(
                f"{method}: "
                f"PSNR={metrics.get('PSNR', float('nan')):.5f} | "
                f"SSIM={metrics.get('SSIM', float('nan')):.5f} | "
                f"LPIPS={metrics.get('LPIPS', float('nan')):.5f}",
                logfile,
            )

    if stats_path.exists():
        log("\nTRAINING STATISTICS:", logfile)

        for line in stats_path.read_text().splitlines():
            log(line, logfile)

    log(
        f"\n[COMPLETED] {exp.name} | " f"Training time: {train_time:.2f}s",
        logfile,
    )


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    selected = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = args.output.resolve() / timestamp
    run_root.mkdir(parents=True, exist_ok=False)

    log_path = run_root / "ablation.log"

    print(f"Experiment directory: {run_root}")
    print(f"Master log: {log_path}")

    with log_path.open("w", buffering=1) as logfile:
        log("=" * 80, logfile)
        log("CoCO-Train Automated Ablation Study", logfile)
        log(f"Date: {datetime.now().isoformat()}", logfile)
        log(f"Source: {args.source.resolve()}", logfile)
        log(f"Iterations: {args.iterations}", logfile)
        log(f"Checkpoints: {args.checkpoints}", logfile)
        log(f"Tight AABB: {args.tight_aabb}", logfile)
        log(f"Lambda opacity: {args.lambda_opacity}", logfile)
        log(f"Selected: {[e.name for e in selected]}", logfile)
        log("=" * 80, logfile)

        for index, exp in enumerate(selected, start=1):
            log(f"\n\nEXPERIMENT {index}/{len(selected)}", logfile)

            try:
                run_experiment(exp, args, run_root, logfile)
            except Exception as error:
                log(f"[EXCEPTION] {exp.name}: {error!r}", logfile)
                continue

        log("\nALL EXPERIMENTS FINISHED.", logfile)

    print("\nDone.")
    print(f"Results: {run_root}")


if __name__ == "__main__":
    main()
