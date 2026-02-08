#!/usr/bin/env python3
"""
Summarize REAP experiment results into a comparison table.

Usage:
    python experiments/summarize_results.py artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/pruned_models/
    
    # Or specific experiments:
    python experiments/summarize_results.py path/to/reap-seed_42-0.25 path/to/reap-seed_42-0.50
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_lm_eval_results(eval_dir: Path) -> Dict[str, float]:
    """Parse lm_eval results from JSON or text table file."""
    metrics = {}
    
    # Try JSON first
    results_file = eval_dir / "lm_eval_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            results = data.get("results", {})
            
            benchmark_mappings = {
                "mmlu": ("mmlu", "acc,none"),
                "hellaswag": ("hellaswag", "acc_norm,none"),
                "boolq": ("boolq", "acc,none"),
                "arc_challenge": ("arc_challenge", "acc_norm,none"),
                "arc_easy": ("arc_easy", "acc_norm,none"),
                "winogrande": ("winogrande", "acc,none"),
                "openbookqa": ("openbookqa", "acc_norm,none"),
                "rte": ("rte", "acc,none"),
            }
            
            for name, (task, metric) in benchmark_mappings.items():
                if task in results and metric in results[task]:
                    metrics[name] = results[task][metric] * 100
            
            if metrics:
                return metrics
        except json.JSONDecodeError:
            pass  # Fall through to text file
    
    # Fallback to text table
    table_file = eval_dir / "lm_eval_results_table.txt"
    if table_file.exists():
        metrics = parse_lm_eval_table(table_file)
    
    return metrics


def parse_lm_eval_table(table_file: Path) -> Dict[str, float]:
    """Parse the lm_eval_results_table.txt format.
    
    Table format: | Task | Version | Filter | n-shot | Metric | dir | Value | sep | Stderr |
    Indices:        1       2         3        4        5       6      7      8      9
    """
    metrics = {}
    
    # Which metric to use for each task
    preferred_metrics = {
        "mmlu": "acc",
        "hellaswag": "acc_norm",
        "boolq": "acc",
        "arc_challenge": "acc_norm",
        "arc_easy": "acc_norm",
        "winogrande": "acc",
        "openbookqa": "acc_norm",
        "rte": "acc",
    }
    
    with open(table_file) as f:
        lines = f.readlines()
    
    current_task = None
    for line in lines:
        if not line.startswith("|"):
            continue
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        
        task = parts[1].strip()
        if task and not task.startswith("-") and task != "Tasks":
            current_task = task
        
        if current_task in preferred_metrics:
            metric_name = parts[5].strip() if len(parts) > 5 else ""
            if metric_name == preferred_metrics[current_task]:
                try:
                    value = float(parts[7].strip())
                    metrics[current_task] = value * 100
                except (ValueError, IndexError):
                    pass
    
    return metrics


def parse_evalplus_results(eval_dir: Path) -> Dict[str, float]:
    """Parse evalplus JSON results for HumanEval and MBPP."""
    metrics = {}
    
    for task in ["humaneval", "mbpp"]:
        # Try direct JSON file first (newer format)
        json_file = eval_dir / f"{task}.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if "pass_at_k" in data:
                    metrics[task] = data["pass_at_k"]["base"]["pass@1"] * 100
                    metrics[f"{task}+"] = data["pass_at_k"]["plus"]["pass@1"] * 100
                continue
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {json_file}: {e}")
        
        # Try evalplus_results subdirectory (older format)
        evalplus_dir = eval_dir / "evalplus_results"
        if evalplus_dir.exists():
            task_dir = evalplus_dir / task
            if task_dir.exists():
                json_files = list(task_dir.glob("*.json"))
                if json_files:
                    try:
                        with open(json_files[0]) as f:
                            data = json.load(f)
                        if "pass_at_k" in data:
                            metrics[task] = data["pass_at_k"]["base"]["pass@1"] * 100
                            metrics[f"{task}+"] = data["pass_at_k"]["plus"]["pass@1"] * 100
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not parse {json_files[0]}: {e}")
    
    return metrics


def get_experiment_name(path: Path) -> str:
    """Extract a short experiment name from the path."""
    name = path.name
    # Extract compression ratio from name like "reap-seed_42-0.25"
    if "-" in name:
        parts = name.split("-")
        for part in parts:
            if "." in part and part.replace(".", "").isdigit():
                ratio = float(part)
                return f"{int(ratio * 100)}% pruned"
    return name


def collect_results(experiment_paths: List[Path]) -> Dict[str, Dict[str, float]]:
    """Collect results from multiple experiments."""
    all_results = {}
    
    for exp_path in experiment_paths:
        eval_dir = exp_path / "eval"
        if not eval_dir.exists():
            print(f"Warning: No eval directory found in {exp_path}")
            continue
        
        exp_name = get_experiment_name(exp_path)
        metrics = {}
        metrics.update(parse_lm_eval_results(eval_dir))
        metrics.update(parse_evalplus_results(eval_dir))
        
        if metrics:
            all_results[exp_name] = metrics
    
    return all_results


def generate_markdown_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate a markdown comparison table."""
    if not results:
        return "No results found."
    
    # Collect all unique benchmarks
    all_benchmarks = set()
    for metrics in results.values():
        all_benchmarks.update(metrics.keys())
    
    # Order benchmarks sensibly
    benchmark_order = [
        "mmlu", "humaneval", "humaneval+", "mbpp", "mbpp+",
        "hellaswag", "boolq", "arc_challenge", "arc_easy",
        "winogrande", "openbookqa", "rte"
    ]
    benchmarks = [b for b in benchmark_order if b in all_benchmarks]
    benchmarks.extend([b for b in sorted(all_benchmarks) if b not in benchmarks])
    
    # Build table
    exp_names = list(results.keys())
    
    # Header
    lines = []
    header = "| Benchmark |" + "|".join(f" {name} " for name in exp_names) + "|"
    separator = "|" + "|".join("-" * (len(name) + 2) for name in ["Benchmark"] + exp_names) + "|"
    lines.append(header)
    lines.append(separator)
    
    # Rows
    for benchmark in benchmarks:
        row = f"| {benchmark} |"
        for exp_name in exp_names:
            value = results[exp_name].get(benchmark)
            if value is not None:
                row += f" {value:.1f}% |"
            else:
                row += " - |"
        lines.append(row)
    
    return "\n".join(lines)


def generate_csv(results: Dict[str, Dict[str, float]]) -> str:
    """Generate CSV output."""
    if not results:
        return ""
    
    all_benchmarks = set()
    for metrics in results.values():
        all_benchmarks.update(metrics.keys())
    benchmarks = sorted(all_benchmarks)
    
    exp_names = list(results.keys())
    
    lines = ["benchmark," + ",".join(exp_names)]
    for benchmark in benchmarks:
        row = [benchmark]
        for exp_name in exp_names:
            value = results[exp_name].get(benchmark)
            row.append(f"{value:.2f}" if value else "")
        lines.append(",".join(row))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize REAP experiment results into comparison tables"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to experiment directories or parent directory containing multiple experiments"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "both"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Collect experiment paths
    experiment_paths = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path does not exist: {path}")
            continue
        
        # Check if this is a parent directory containing multiple experiments
        if (path / "eval").exists():
            # This is an experiment directory
            experiment_paths.append(path)
        else:
            # This might be a parent directory
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir() and (subdir / "eval").exists():
                    experiment_paths.append(subdir)
    
    if not experiment_paths:
        print("No valid experiment directories found.")
        return
    
    print(f"Found {len(experiment_paths)} experiments:")
    for p in experiment_paths:
        print(f"  - {p.name}")
    print()
    
    results = collect_results(experiment_paths)
    
    output_lines = []
    
    if args.format in ["markdown", "both"]:
        output_lines.append("## Results Comparison\n")
        output_lines.append(generate_markdown_table(results))
        output_lines.append("")
    
    if args.format in ["csv", "both"]:
        if args.format == "both":
            output_lines.append("\n## CSV Format\n```csv")
        output_lines.append(generate_csv(results))
        if args.format == "both":
            output_lines.append("```")
    
    output = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
