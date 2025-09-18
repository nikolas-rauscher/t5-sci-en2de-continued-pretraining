#!/usr/bin/env python3
"""
Generate per-family progression evaluation configs (MMLU EN) for all checkpoints.

For each Hydra run root (by default: all families marked [still running] in
docs/t5_pretraining_runs_report.md), this script discovers checkpoints across
timestamped runs and writes a single Hydra experiment YAML enumerating models
for those checkpoints. You can control how densely to sample steps.

Typical usage:
  # Generate configs for all [still running] families, sampling every 25k steps
  python scripts/generate_progression_eval_configs.py

  # Include ALL checkpoints (heavy!)
  python scripts/generate_progression_eval_configs.py --all

  # Only a specific family (glob allowed)
  python scripts/generate_progression_eval_configs.py --filter pretraining_logs_lr_001_bugfixed_clean_restart

  # Sample every 50k steps and include best checkpoints
  python scripts/generate_progression_eval_configs.py --step-interval 50000 --include-best

The resulting configs live under configs/experiment/progression_<family>_<ts>.yaml
Run them on the cluster with:
  sbatch jobs/flexible_eval_24h.sh progression_<family>_<ts>
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_REPORT = PROJECT_ROOT / "docs/t5_pretraining_runs_report.md"
EXPERIMENT_DIR = PROJECT_ROOT / "configs/experiment"


def parse_still_running_roots(report_path: Path) -> list[Path]:
    text = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    roots: list[Path] = []

    i = 0
    while i < len(text):
        line = text[i]
        if (line.startswith("### ") or line.startswith("#### ")) and "[still running]" in line:
            j = i + 1
            while j < len(text) and not (text[j].startswith("### ") or text[j].startswith("#### ")):
                l = text[j]
                if "Hydra root" in l and "`" in l:
                    try:
                        rel = l.split("`")[1].strip()
                        if rel:
                            roots.append(PROJECT_ROOT / rel)
                            break
                    except Exception:
                        pass
                j += 1
            i = j
        else:
            i += 1

    # Deduplicate preserving order
    uniq, seen = [], set()
    for p in roots:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def list_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d.name)]
    )


def discover_checkpoints(run_dirs: Iterable[Path]) -> list[Path]:
    ckpts: list[Path] = []
    for rd in run_dirs:
        ckpt_dir = rd / "checkpoints"
        if not ckpt_dir.exists():
            continue
        # steps and best folders
        for sub in (ckpt_dir / "steps", ckpt_dir / "best"):
            if sub.exists():
                for c in sorted(sub.glob("*.ckpt")):
                    if c.name == "last.ckpt":
                        continue  # skip pointer files lacking step info
                    ckpts.append(c)
    return ckpts


def extract_step(path: Path) -> int:
    name = path.name
    m = re.search(r"step-(\d+)-val_ppl-", name)
    if m:
        return int(m.group(1))
    m = re.search(r"step-step=(\d+)\.ckpt", name)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch-\d+-step-(\d+)\.ckpt", name)
    if m:
        return int(m.group(1))
    return -1


def is_best_ckpt(path: Path) -> bool:
    return path.parent.name == "best" or "-val_ppl-" in path.name


def sample_checkpoints(ckpts: list[Path], include_best: bool, step_interval: int | None, include_all: bool) -> list[Path]:
    # First, deduplicate by step number - prefer best checkpoints over regular ones
    by_step: dict[int, Path] = {}
    for c in ckpts:
        step = extract_step(c)
        if step < 0:
            continue
        # If we already have this step, prefer the best checkpoint
        if step in by_step:
            if is_best_ckpt(c) and not is_best_ckpt(by_step[step]):
                by_step[step] = c
        else:
            by_step[step] = c
    
    # Now work with deduplicated checkpoints
    unique_ckpts = list(by_step.values())
    
    if include_all:
        return sorted(unique_ckpts, key=extract_step)

    # Filter and sample
    selected: list[Path] = []
    seen_steps = set()

    # Always include best checkpoints if requested
    if include_best:
        bests = [c for c in unique_ckpts if is_best_ckpt(c)]
        bests.sort(key=extract_step)
        selected.extend(bests)
        seen_steps.update(extract_step(c) for c in bests)

    # Add step checkpoints at requested interval
    if step_interval and step_interval > 0:
        steps = sorted(unique_ckpts, key=extract_step)
        last_included = -step_interval
        for c in steps:
            s = extract_step(c)
            if s - last_included >= step_interval and s not in seen_steps:
                selected.append(c)
                last_included = s
                seen_steps.add(s)

    return sorted(selected, key=extract_step)


def family_name_from_root(root: Path) -> str:
    try:
        rel = root.relative_to(PROJECT_ROOT)
        return rel.parts[0]
    except Exception:
        return root.name


def write_experiment_yaml(family: str, ckpts: list[Path]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"progression_{family}_{ts}"
    out_path = EXPERIMENT_DIR / f"{exp_name}.yaml"

    lines: list[str] = []
    lines.append("# @package _global_")
    lines.append("")
    lines.append(f"experiment_name: \"{exp_name}\"")
    lines.append(f"description: \"Progression MMLU EN — {family} (auto-generated)\"")
    lines.append("")
    lines.append("models:")
    # Use actual family name for run label
    run_label = family

    for idx, c in enumerate(ckpts, start=1):
        try:
            rel = c.relative_to(PROJECT_ROOT)
        except Exception:
            rel = c
        step_k = extract_step(c)
        step_k = step_k // 1000 if step_k >= 0 else 0
        # Name pattern compatible with analyzer: <run_type>_stepNN_YYk
        name = f"{run_label}_step{idx:02d}_{step_k}k"
        lines.append(f"  - source_path: \"{rel}\"")
        lines.append(f"    name: \"{name}\"")
    lines.append("")
    lines.append("benchmarks:")
    lines.append("  - name: \"mmlu\"")
    lines.append("    shots: [0]")
    lines.append("    seed: 42")
    lines.append("    device: \"cuda\"")
    lines.append("    batch_size: \"auto\"")
    lines.append("")
    lines.append("logger:")
    lines.append("  wandb:")
    lines.append("    project: \"BA-T5-Evaluation\"")
    lines.append("    entity: \"nikolas-rauscher-dfki\"")
    lines.append("    group: \"Progression-MMLU-EN\"")
    lines.append(f"    name: \"{exp_name}\"")
    lines.append("    tags: [\"mmlu\", \"progression\", \"en\", \"auto\"]")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Include ALL checkpoints (warning: heavy)")
    ap.add_argument("--include-best", action="store_true", help="Always include best checkpoints")
    ap.add_argument("--step-interval", type=int, default=25000, help="Sample step checkpoints every N steps (ignored with --all)")
    ap.add_argument("--filter", type=str, default=None, help="Only include families whose root matches this substring")
    ap.add_argument("--root", type=str, default=None, help="Directly specify a root path (e.g., pretraining_logs_lr_001_OPTIMIZED_clean_restart_v2/train/runs)")
    args = ap.parse_args()

    # If --root is provided, use it directly
    if args.root:
        root_path = Path(args.root)
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
        # Handle both full path and path without /train/runs
        if not root_path.name == "runs":
            root_path = root_path / "train" / "runs"
        roots = [root_path]
    else:
        roots = parse_still_running_roots(RUNS_REPORT)
        if args.filter:
            roots = [r for r in roots if args.filter in str(r)]
    
    if not roots:
        print("No matching families found.")
        return

    for root in roots:
        family = family_name_from_root(root)
        run_dirs = list_run_dirs(root)
        ckpts = discover_checkpoints(run_dirs)
        if not ckpts:
            print(f"[skip] {family}: no checkpoints found")
            continue

        selected = sample_checkpoints(ckpts, include_best=args.include_best, step_interval=args.step_interval, include_all=args.all)
        if not selected:
            print(f"[skip] {family}: selection empty")
            continue
        exp_path = write_experiment_yaml(family, selected)
        print(f"Generated: {exp_path.relative_to(PROJECT_ROOT)}  (models={len(selected)})")
        print(f"  sbatch jobs/flexible_eval_24h.sh {exp_path.stem}")


if __name__ == "__main__":
    main()
