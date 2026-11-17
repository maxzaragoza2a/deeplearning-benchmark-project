"""Short runs to choose the optimizer and learning rate of PRWN + ResNet-12, on `valid` only.

Each candidate is a shortened run of scripts/train_harmonized.py (same data, same seed,
hence the same training and validation episodes) with `--skip-test`: the test episodes
are never looked at, so choosing the best candidate does not leak the test set.

Candidates: AdamW 1e-3 (the reference run), SGD with the exact NCA settings (lr 0.1,
x0.1 at 70%), and SGD with a cosine schedule at decreasing learning rates.

The ranking is written to results/sweep_optimizer/summary.csv. Then set the winning lr
(and schedule) in configs/harmonized/prwn_resnet12_sgd.yaml and run the full 20,000 steps.

Examples:
    python scripts/sweep_optimizer.py --device cuda
    python scripts/sweep_optimizer.py --device cuda --steps 4000 --only sgd_lr0.03 sgd_lr0.01
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "harmonized" / "prwn_resnet12_sgd.yaml"
SWEEP_DIR = "sweep_optimizer"

# name -> overrides of the `train` section (on top of prwn_resnet12_sgd.yaml)
CANDIDATES = {
    "adamw_lr1e-3": {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "lr_schedule": "cosine"},
    "sgd_lr0.1_step": {"optimizer": "sgd", "lr": 0.1, "weight_decay": 5e-4, "lr_schedule": "step"},  # exact NCA settings
    "sgd_lr0.1": {"optimizer": "sgd", "lr": 0.1, "weight_decay": 5e-4, "lr_schedule": "cosine"},
    "sgd_lr0.03": {"optimizer": "sgd", "lr": 0.03, "weight_decay": 5e-4, "lr_schedule": "cosine"},
    "sgd_lr0.01": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 5e-4, "lr_schedule": "cosine"},
    "sgd_lr0.003": {"optimizer": "sgd", "lr": 0.003, "weight_decay": 5e-4, "lr_schedule": "cosine"},
    # Added after the first sweep: 0.003 was the best and the lowest tested lr, so the
    # optimum may lie below it.
    "sgd_lr0.001": {"optimizer": "sgd", "lr": 0.001, "weight_decay": 5e-4, "lr_schedule": "cosine"},
    "sgd_lr0.0003": {"optimizer": "sgd", "lr": 0.0003, "weight_decay": 5e-4, "lr_schedule": "cosine"},
}


def run_candidate(name: str, overrides: dict, steps: int, device: str, results_dir: Path,
                  extra_sets: list[str]) -> dict:
    """Trains one candidate for `steps` steps and returns its validation summary."""
    sets = [f"run_name={SWEEP_DIR}/{name}", f"train.n_steps={steps}", f"results_dir={results_dir.as_posix()}"]
    sets += [f"train.{key}={value}" for key, value in overrides.items()] + extra_sets
    command = [sys.executable, str(ROOT / "scripts" / "train_harmonized.py"), "--config", str(CONFIG),
               "--device", device, "--skip-test"]
    for item in sets:
        command += ["--set", item]

    run_dir = results_dir / SWEEP_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").unlink(missing_ok=True)  # never report a previous run as this one
    start = time.time()
    with open(run_dir / "train.log", "w", encoding="utf-8") as log:
        returncode = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    minutes = round((time.time() - start) / 60, 1)
    if returncode != 0:
        tail = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        return {"candidate": name, **overrides, "minutes": minutes,
                "status": f"failed: {tail[0] if tail else returncode}"}
    return read_candidate(name, overrides, run_dir, minutes)


def read_candidate(name: str, overrides: dict, run_dir: Path, minutes: float | str = "") -> dict:
    """Validation summary of a finished candidate, read from its metrics.json and history.csv."""
    row = {"candidate": name, **overrides, "minutes": minutes}
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        tail = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        return {**row, "status": f"failed: {tail[0] if tail else returncode}"}

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    with open(run_dir / "history.csv", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    last = history[-1]
    return {
        **row,
        "status": "ok",
        "best_val": round(metrics["best_val"]["accuracy"], 2),
        "best_val_ci95": round(metrics["best_val"]["ci95"], 2),
        "best_step": metrics["best_val"]["step"],
        "last_val": round(float(last["val_accuracy"]), 2),
        "last_train_acc": round(100 * float(last["train_accuracy"]), 1),
        "last_loss": round(float(last["train_loss"]), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=4000, help="training steps per candidate")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--only", nargs="+", choices=sorted(CANDIDATES), help="run a subset of the candidates")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="extra override passed to every candidate (quick checks only, e.g. limit_per_class=2)")
    args = p.parse_args()
    args.results_dir = args.results_dir.resolve()

    names = args.only or list(CANDIDATES)
    print(f"{len(names)} candidates x {args.steps} steps, selection on valid only (test never run)")
    rows = []
    for name in names:
        print(f"  {name} ...", end=" ", flush=True)
        row = run_candidate(name, CANDIDATES[name], args.steps, args.device, args.results_dir, args.set)
        rows.append(row)
        result = f"best val {row['best_val']} % at step {row['best_step']}" if row["status"] == "ok" else row["status"]
        print(f"{result} ({row['minutes']} min)", flush=True)

    # Candidates finished in an earlier sweep (e.g. before --only) join the ranking.
    for name, overrides in CANDIDATES.items():
        run_dir = args.results_dir / SWEEP_DIR / name
        if name not in names and (run_dir / "metrics.json").exists():
            row = read_candidate(name, overrides, run_dir)
            if row.get("best_step") is not None and run_dir.joinpath("history.csv").exists():
                rows.append({**row, "candidate": f"{name} (earlier)"})

    ranked = sorted(rows, key=lambda r: r.get("best_val", -1), reverse=True)
    out = args.results_dir / SWEEP_DIR / "summary.csv"
    fields = ["candidate", "optimizer", "lr", "lr_schedule", "weight_decay", "status", "best_val",
              "best_val_ci95", "best_step", "last_val", "last_train_acc", "last_loss", "minutes"]
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(ranked)

    print(f"\nRanking on valid (5-way 1-shot, {args.steps} steps):")
    print(f"  {'candidate':<24} {'best val':>9} {'+/-':>5} {'step':>6} {'last val':>9} {'train acc':>10}")
    for row in ranked:
        if row["status"] == "ok":
            print(f"  {row['candidate']:<24} {row['best_val']:>8.2f}% {row['best_val_ci95']:>5.2f} "
                  f"{row['best_step']:>6} {row['last_val']:>8.2f}% {row['last_train_acc']:>9.1f}%")
        else:
            print(f"  {row['candidate']:<24} {row['status']}")
    print(f"\nWritten to {out}")
    print("Candidates within about one CI of the best are tied: prefer the simplest (cosine) among them.")


if __name__ == "__main__":
    main()
