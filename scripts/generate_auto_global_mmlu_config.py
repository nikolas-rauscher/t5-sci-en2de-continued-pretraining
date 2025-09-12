#!/usr/bin/env python3
"""
Generate a Global MMLU eval config from all families marked [still running]
in docs/t5_pretraining_runs_report.md. For each Hydra run root, the script
selects the best checkpoint if available (best/last.ckpt), otherwise falls back
to the latest step checkpoint (steps/last.ckpt), and finally to the max step-*.ckpt.

Outputs a Hydra experiment YAML under configs/experiment/ with models and two
benchmarks (global MMLU EN/DE, zero-shot), ready to be used via:
  sbatch jobs/flexible_eval_24h.sh <generated_experiment_name>

Usage:
  python scripts/generate_auto_global_mmlu_config.py \
    --runs-report docs/t5_pretraining_runs_report.md \
    --output configs/experiment/auto_global_mmlu_<timestamp>.yaml

Optional flags:
  --prefer {best,latest}  # default: best
  --shots 0               # default: 0
  --include-en --include-de  # both enabled by default
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_heading(line: str) -> bool:
    """Return True if the line looks like a Markdown heading (#, ##, ###, ...)."""
    s = line.lstrip()
    return s.startswith("#")


def parse_still_running_roots(report_path: Path) -> list[Path]:
    text = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    roots: list[Path] = []

    i = 0
    while i < len(text):
        line = text[i]
        # Accept any heading level (###, ####, etc.) and match marker case-insensitively
        if _is_heading(line) and "[still running]" in line.lower():
            # Scan forward until the next heading or end, looking for a line containing "Hydra root"
            j = i + 1
            found = False
            while j < len(text) and not _is_heading(text[j]):
                l = text[j]
                if "Hydra root" in l and "`" in l:
                    # Extract first backtick-quoted path
                    try:
                        rel = l.split("`")[1].strip()
                        if rel:
                            roots.append(PROJECT_ROOT / rel)
                            found = True
                            break
                    except Exception:
                        pass
                j += 1
            # Advance to next section block
            i = j if found else i + 1
        else:
            i += 1

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for p in roots:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def list_timestamped_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    # Expect directories like YYYY-MM-DD_HH-MM-SS
    dirs = [d for d in run_root.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d.name)]
    # Sort chronologically by folder name (lexicographic works for this format)
    return sorted(dirs)


def find_checkpoint_for_run_dir(run_dir: Path, prefer: str) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    best_last = ckpt_dir / "best" / "last.ckpt"
    steps_last = ckpt_dir / "steps" / "last.ckpt"

    if prefer == "best":
        if best_last.exists():
            return best_last
        if steps_last.exists():
            return steps_last
    else:  # latest preference
        if steps_last.exists():
            return steps_last
        if best_last.exists():
            return best_last

    # Fallback: choose the highest step-* checkpoint under best/ or steps/
    candidates = list((ckpt_dir / "best").glob("step-*.ckpt")) + list((ckpt_dir / "steps").glob("step-*.ckpt"))
    if candidates:
        def extract_step(p: Path) -> int:
            # Try patterns: step-123456-val_ppl-1.234.ckpt or step-step=123456.ckpt or epoch-1-step-123456.ckpt
            m = re.search(r"step-(\d+)-val_ppl-", p.name)
            if m:
                return int(m.group(1))
            m = re.search(r"step-step=(\d+)\.ckpt", p.name)
            if m:
                return int(m.group(1))
            m = re.search(r"epoch-\d+-step-(\d+)\.ckpt", p.name)
            if m:
                return int(m.group(1))
            return -1

        candidates.sort(key=extract_step)
        return candidates[-1]

    return None


def pick_checkpoint_for_root(root: Path, prefer: str) -> Path | None:
    run_dirs = list_timestamped_run_dirs(root)
    # Search newest to oldest
    for run_dir in reversed(run_dirs):
        ckpt = find_checkpoint_for_run_dir(run_dir, prefer)
        if ckpt and ckpt.exists():
            return ckpt
    return None


def build_yaml(models: list[tuple[str, str]], include_en: bool, include_de: bool, shots: list[int], exp_name: str) -> str:
    # models is list of (relative_source_path, name)
    lines: list[str] = []
    lines.append("# @package _global_")
    lines.append("")
    lines.append(f"experiment_name: \"{exp_name}\"")
    lines.append("description: \"Auto-generated: Global MMLU EN/DE zero-shot for [still running] families\"")
    lines.append("")
    lines.append("models:")
    for rel_path, name in models:
        lines.append(f"  - source_path: \"{rel_path}\"")
        lines.append(f"    name: \"{name}\"")
    lines.append("")
    lines.append("benchmarks:")
    if include_en:
        lines.append("  # Global MMLU English - Zero shot")
        lines.append("  - name: \"global_mmlu_en\"")
        lines.append("    tasks: [\"global_mmlu_full_en\"]")
        shots_str = ", ".join(str(s) for s in shots)
        lines.append(f"    shots: [{shots_str}]")
        lines.append("    seed: 42")
        lines.append("    device: \"cuda\"")
        lines.append("    batch_size: \"auto\"")
        lines.append("")
    if include_de:
        lines.append("  # Global MMLU German - Zero shot")
        lines.append("  - name: \"global_mmlu_de\"")
        lines.append("    tasks: [\"global_mmlu_full_de\"]")
        shots_str = ", ".join(str(s) for s in shots)
        lines.append(f"    shots: [{shots_str}]")
        lines.append("    seed: 42")
        lines.append("    device: \"cuda\"")
        lines.append("    batch_size: \"auto\"")
        lines.append("")

    # Minimal W&B logger (can be edited later)
    lines.append("logger:")
    lines.append("  wandb:")
    lines.append("    project: \"BA-T5-CrossLingual\"")
    lines.append("    entity: \"nikolas-rauscher-dfki\"")
    lines.append("    group: \"Auto-Global-MMLU\"")
    lines.append(f"    name: \"{exp_name}\"")
    lines.append("    tags: [\"global-mmlu\", \"auto\", \"still-running\", \"zero-shot\"]")
    lines.append("")

    return "\n".join(lines) + "\n"


def _extract_step_from_filename(fname: str) -> int:
    """Return step from common checkpoint filename patterns or -1 if unknown.

    Supports:
      - step-123456-val_ppl-....ckpt
      - step-step=123456.ckpt (Lightning style)
      - epoch-1-step-123456.ckpt
    """
    import re as _re
    m = _re.search(r"step-(\d+)-val_ppl-", fname)
    if m:
        return int(m.group(1))
    m = _re.search(r"step-step=(\d+)\.ckpt", fname)
    if m:
        return int(m.group(1))
    m = _re.search(r"epoch-\d+-step-(\d+)\.ckpt", fname)
    if m:
        return int(m.group(1))
    return -1


def _derive_step_for_ckpt_path(ckpt: Path) -> int:
    """Best-effort step discovery for a selected ckpt path.

    If the filename encodes the step, extract it. If it is a last.ckpt, scan the
    sibling directory for the highest step-*.ckpt and use that step number.
    Returns -1 if no step can be inferred.
    """
    step = _extract_step_from_filename(ckpt.name)
    if step > 0:
        return step
    # last.ckpt case: look for step-* siblings
    if ckpt.name == "last.ckpt" and ckpt.parent.exists():
        candidates = list(ckpt.parent.glob("step-*.ckpt"))
        if candidates:
            candidates.sort(key=lambda p: _extract_step_from_filename(p.name))
            s = _extract_step_from_filename(candidates[-1].name)
            return s if s > 0 else -1
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-report", default=str(PROJECT_ROOT / "docs/t5_pretraining_runs_report.md"))
    ap.add_argument("--prefer", choices=["best", "latest"], default="best")
    ap.add_argument("--shots", type=int, nargs="*", default=[0])
    # Language selection flags (default: include both EN and DE)
    ap.add_argument("--include-en", dest="include_en", action="store_true")
    ap.add_argument("--include-de", dest="include_de", action="store_true")
    ap.add_argument("--no-en", dest="include_en", action="store_false")
    ap.add_argument("--no-de", dest="include_de", action="store_false")
    ap.add_argument("--en-only", action="store_true", help="Include only English (equivalent to --no-de)")
    ap.set_defaults(include_en=True, include_de=True)
    ap.add_argument("--output", default=None, help="Output YAML path under configs/experiment/")
    args = ap.parse_args()

    runs_report = Path(args.runs_report)
    if not runs_report.exists():
        print(f"Runs report not found: {runs_report}", file=sys.stderr)
        sys.exit(1)

    # Handle convenience flag
    if args.en_only:
        args.include_en, args.include_de = True, False

    roots = parse_still_running_roots(runs_report)
    if not roots:
        print("No [still running] Hydra roots found.", file=sys.stderr)
        sys.exit(2)

    models: list[tuple[str, str]] = []
    for root in roots:
        # Compute a friendly family name from root path
        try:
            # Expect .../<family>/train/runs
            parts = root.relative_to(PROJECT_ROOT).parts
        except Exception:
            parts = root.parts
        family = parts[0] if parts else root.name

        ckpt = pick_checkpoint_for_root(root, args.prefer)
        if not ckpt:
            print(f"Warning: No checkpoint found under {root}")
            continue

        # Prefer relative paths in config
        rel_ckpt: str
        try:
            rel_ckpt = str(ckpt.relative_to(PROJECT_ROOT))
        except Exception:
            rel_ckpt = str(ckpt)

        # Derive a display name indicating family, choice and (if available) step count
        choice_tag = "best" if "best/" in rel_ckpt else ("last" if rel_ckpt.endswith("last.ckpt") else "step")
        step_num = _derive_step_for_ckpt_path(ckpt)
        if step_num and step_num > 0:
            model_name = f"{family}-{choice_tag}-{step_num//1000}k"
        else:
            model_name = f"{family}-{choice_tag}"
        models.append((rel_ckpt, model_name))

    if not models:
        print("No checkpoints discovered across [still running] families.", file=sys.stderr)
        sys.exit(3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_short = f"auto_global_mmlu_still_running_{timestamp}"
    out_path = Path(args.output) if args.output else (PROJECT_ROOT / "configs/experiment" / f"{exp_short}.yaml")
    exp_name = out_path.stem

    yaml_text = build_yaml(models, args.include_en, args.include_de, args.shots, exp_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text)

    print("Generated eval config:")
    print(out_path)
    print("Models:")
    for rel, name in models:
        print(f"- {name}: {rel}")

    print("\nSubmit with:")
    print(f"sbatch jobs/flexible_eval_24h.sh {exp_name}")


if __name__ == "__main__":
    main()
