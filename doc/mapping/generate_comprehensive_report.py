#!/usr/bin/env python3
"""
Comprehensive verification report generator.
Merges results from all verification batches into a single analysis report.
"""

import glob
import json
import os
from datetime import datetime


def load_json_results(output_dir: str) -> dict:
    """Load all timestamped JSON result files."""
    pattern = os.path.join(output_dir, "verification_results_*.json")
    files = sorted(glob.glob(pattern))

    all_results = {}
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_results.update(data)

    return all_results


def analyze_results(all_results: dict) -> dict:
    """Analyze all verification results."""
    analysis = {
        "total_verified": 0,
        "by_status": {},
        "by_batch": {},
        "alias_candidates": [],
        "kernel_only": [],
        "truly_missing": [],
        "classification_issues": [],
        "pytorch_trace_failures": [],
    }

    for batch_name, batch_result in all_results.items():
        batch_stats = {
            "total": batch_result.get("total", 0),
            "statuses": {},
        }

        for r in batch_result.get("results", []):
            analysis["total_verified"] += 1
            status = r.get("verification_status", "unknown")
            batch_stats["statuses"][status] = (
                batch_stats["statuses"].get(status, 0) + 1
            )
            analysis["by_status"][status] = (
                analysis["by_status"].get(status, 0) + 1
            )

            # Collect alias candidates
            for cand in r.get("alias_candidates", []):
                key = (cand["torch_api"], cand["paddle_api"])
                if key not in [
                    (a["torch_api"], a["paddle_api"])
                    for a in analysis["alias_candidates"]
                ]:
                    analysis["alias_candidates"].append(cand)

            # Collect kernel_only
            if status == "kernel_only":
                analysis["kernel_only"].append(r)

            # Collect truly_missing
            if status == "truly_missing":
                analysis["truly_missing"].append(r)

            # Collect PyTorch trace failures
            pytorch = r.get("pytorch", {})
            if not pytorch or not pytorch.get("declaration", {}).get("found"):
                analysis["pytorch_trace_failures"].append(r["op_name"])

        analysis["by_batch"][batch_name] = batch_stats

    return analysis


def generate_comprehensive_markdown(
    all_results: dict, analysis: dict, output_path: str
):
    """Generate comprehensive Markdown report."""
    lines = []
    lines.append("# API 映射综合验证报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall summary
    lines.append("## 总体概览")
    lines.append("")
    total = analysis["total_verified"]
    lines.append(f"- **总验证 API 数**: {total}")
    for status, count in sorted(analysis["by_status"].items()):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"- **{status}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Batch summary
    lines.append("## 各批次验证结果")
    lines.append("")
    lines.append("| 批次 | 总数 | 状态分布 |")
    lines.append("|------|------|----------|")
    for batch_name, stats in sorted(analysis["by_batch"].items()):
        status_str = ", ".join(
            f"{s}: {c}" for s, c in sorted(stats["statuses"].items())
        )
        lines.append(f"| {batch_name} | {stats['total']} | {status_str} |")
    lines.append("")

    # Alias candidates
    aliases = analysis["alias_candidates"]
    if aliases:
        lines.append("## 别名映射候选汇总")
        lines.append("")
        lines.append("| PyTorch API | Paddle API | 规则 | 置信度 |")
        lines.append("|-------------|-----------|------|--------|")
        for cand in sorted(aliases, key=lambda x: x["torch_api"]):
            lines.append(
                f"| `at::{cand['torch_api']}` | `paddle::experimental::{cand['paddle_api']}` | {cand['rule']} | {cand['confidence']} |"
            )
        lines.append("")
        lines.append(f"**总计发现 {len(aliases)} 个别名候选**")
        lines.append("")

    # Kernel-only candidates
    kernel_only = analysis["kernel_only"]
    if kernel_only:
        lines.append("## Kernel 已注册但未暴露到 api.h")
        lines.append("")
        lines.append("| PyTorch API | Paddle Kernel 文件 | CPU | GPU |")
        lines.append("|-------------|-------------------|-----|-----|")
        for r in kernel_only:
            op = r["op_name"]
            kreg = r.get("paddle", {}).get("kernel_registration", {})
            cpu = "是" if kreg.get("cpu_registered") else "否"
            gpu = "是" if kreg.get("gpu_registered") else "否"
            kfile = kreg.get("cpu_file", "") or kreg.get("gpu_file", "")
            lines.append(f"| `at::{op}` | {kfile} | {cpu} | {gpu} |")
        lines.append("")
        lines.append(f"**总计发现 {len(kernel_only)} 个 kernel_only 候选**")
        lines.append("")

    # Classification issues
    issues = analysis["classification_issues"]
    if issues:
        lines.append("## 映射表分类问题")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    # PyTorch trace failures
    failures = analysis["pytorch_trace_failures"]
    if failures:
        lines.append("## PyTorch 追踪失败")
        lines.append("")
        lines.append(", ".join(f"`at::{f}`" for f in failures))
        lines.append("")

    # Batch details
    lines.append("## 详细批次结果")
    lines.append("")
    for batch_name, batch_result in sorted(all_results.items()):
        lines.append(f"### {batch_name}")
        lines.append("")
        lines.append(f"**总数**: {batch_result.get('total', 0)}")
        lines.append("")

        # Non-standard status APIs
        interesting = [
            r
            for r in batch_result.get("results", [])
            if r.get("verification_status")
            not in ("verified_compat", "verified_api_h_only", "truly_missing")
            or r.get("notes")
            or r.get("alias_candidates")
        ]

        if interesting:
            lines.append("**需关注的 API**:")
            lines.append("")
            lines.append("| API | 当前状态 | 验证状态 | 备注 | 别名候选 |")
            lines.append("|-----|---------|---------|------|----------|")
            for r in interesting:
                op = r["op_name"]
                vstatus = r.get("verification_status", "")
                notes = "; ".join(r.get("notes", []))[:80]
                alias_str = ", ".join(
                    f"{c['paddle_api']}({c['confidence']})"
                    for c in r.get("alias_candidates", [])
                )[:60]
                lines.append(
                    f"| `at::{op}` | - | {vstatus} | {notes} | {alias_str} |"
                )
            lines.append("")
        else:
            lines.append("无需关注")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Comprehensive report saved to: {output_path}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "verification_output")
    output_path = os.path.join(
        output_dir,
        f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
    )

    print("Loading verification results...")
    all_results = load_json_results(output_dir)
    print(f"  Loaded {len(all_results)} batches")

    print("Analyzing results...")
    analysis = analyze_results(all_results)

    print("Generating comprehensive report...")
    generate_comprehensive_markdown(all_results, analysis, output_path)

    print("\nSummary:")
    print(f"  Total APIs verified: {analysis['total_verified']}")
    for status, count in sorted(analysis["by_status"].items()):
        print(f"    {status}: {count}")
    print(f"  Alias candidates found: {len(analysis['alias_candidates'])}")
    print(f"  Kernel-only candidates: {len(analysis['kernel_only'])}")


if __name__ == "__main__":
    main()
