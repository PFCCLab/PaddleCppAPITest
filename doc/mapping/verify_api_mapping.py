#!/usr/bin/env python3
"""
API Mapping Verification Script
Based on Step 2-1 methodology: trace PyTorch implementation chain,
trace Paddle implementation chain, compare and verify mapping accuracy.

Usage:
    python verify_api_mapping.py --batch P0_exact_match
    python verify_api_mapping.py --op abs
    python verify_api_mapping.py --all
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add verify package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verify"))

from alias_detector import AliasDetector
from paddle_tracer import PaddleTracer
from pytorch_tracer import PyTorchTracer

# Default paths: prefer environment variables, fallback to hard-coded defaults
_torch_dir = os.environ.get("TORCH_DIR", r"D:/Lenovo/libtorch")
_paddle_root = os.environ.get("PADDLE_ROOT", r"D:/Lenovo/Paddle")
_pytorch_root = os.environ.get("PYTORCH_ROOT", r"D:/Lenovo/pytorch")

DEFAULT_CONFIG = {
    "libtorch_ops_dir": os.path.join(_torch_dir, "include", "ATen", "ops"),
    "pytorch_src_dir": _pytorch_root,
    "paddle_src_dir": _paddle_root,
    "paddle_api_h": os.path.join(
        _paddle_root, "paddle", "phi", "api", "include", "api.h"
    ),
    "paddle_compat_dir": os.path.join(
        _paddle_root, "paddle", "phi", "api", "include", "compat", "ATen", "ops"
    ),
    "mapping_file": os.path.join(
        os.path.dirname(__file__), "cpp_api_mapping_cn.md"
    ),
}


def parse_mapping_file(mapping_file: str) -> dict:
    """Parse existing mapping markdown to extract categorized ops."""
    with open(mapping_file, "r", encoding="utf-8") as f:
        content = f.read()

    mapping = {}
    current_category = None

    for line in content.split("\n"):
        # Detect category headers like "### 1. API 完全一致"
        cat_match = re.search(r"### \d+\.\s*(.+)", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        # Extract API entries: | 1 | `at::op_name` | ... |
        if current_category and line.startswith("|") and "at::" in line:
            op_match = re.search(r"`at::(\w+)`", line)
            if op_match:
                op_name = op_match.group(1)
                if current_category not in mapping:
                    mapping[current_category] = []
                if op_name not in mapping[current_category]:
                    mapping[current_category].append(op_name)

    return mapping


def create_batches(mapping_data: dict) -> dict:
    """Create prioritized verification batches."""
    batches = {}

    # P0: API 完全一致
    batches["P0_exact_match"] = mapping_data.get("API 完全一致", [])

    # P1: 仅参数名不一致
    batches["P1_name_diff"] = mapping_data.get("仅参数名不一致", [])

    # P2: Other difference categories
    p2_categories = [
        "仅 API 调用方式不一致",
        "paddle 参数更多",
        "torch 参数更多",
        "输入参数用法不一致",
        "输入参数类型不一致",
        "返回参数类型不一致",
        "参数默认值不一致",
    ]
    for cat in p2_categories:
        ops = mapping_data.get(cat, [])
        if ops:
            key = (
                f"P2_{cat.replace(' ', '_').replace('(', '').replace(')', '')}"
            )
            batches[key] = ops

    # P3: API aliases
    batches["P3_alias"] = mapping_data.get("API 别名", [])

    # P4: Missing (largest batch)
    batches["P4_missing"] = mapping_data.get("功能缺失", [])

    # P5: Semantic mismatch
    batches["P5_semantic"] = mapping_data.get("语义差异", [])

    return batches


class APIVerifier:
    """Main API mapping verifier."""

    def __init__(self, config: dict):
        self.config = config
        self.pytorch_tracer = PyTorchTracer(
            config["libtorch_ops_dir"], config.get("pytorch_src_dir")
        )
        self.paddle_tracer = PaddleTracer(
            config["paddle_api_h"], config["paddle_src_dir"]
        )
        self.alias_detector = AliasDetector.from_paddle_tracer(
            self.paddle_tracer
        )

    def verify_op(self, op_name: str) -> dict:
        """Verify a single API."""
        result = {
            "op_name": op_name,
            "pytorch": None,
            "paddle": None,
            "alias_candidates": [],
            "verification_status": "pending",
            "notes": [],
        }

        # Trace PyTorch side
        try:
            result["pytorch"] = self.pytorch_tracer.trace(op_name)
        except Exception as e:
            result["notes"].append(f"PyTorch trace error: {e}")

        # Trace Paddle side
        try:
            result["paddle"] = self.paddle_tracer.trace(op_name)
        except Exception as e:
            result["notes"].append(f"Paddle trace error: {e}")

        # Discover alias candidates
        try:
            result["alias_candidates"] = self.alias_detector.discover(op_name)
        except Exception as e:
            result["notes"].append(f"Alias detection error: {e}")

        # Determine verification status
        result["verification_status"] = self._determine_status(result)

        return result

    def _determine_status(self, result: dict) -> str:
        """Determine the verification status of an API."""
        paddle = result.get("paddle", {})
        pytorch = result.get("pytorch", {})

        has_pytorch = pytorch and pytorch.get("declaration", {}).get("found")
        has_paddle_api = (
            paddle.get("api_declaration", {}).get("found", False)
            if paddle
            else False
        )
        has_paddle_compat = (
            paddle.get("compat_layer", {}).get("found", False)
            if paddle
            else False
        )
        has_paddle_kernel = (
            paddle.get("kernel_registration", {}).get("found", False)
            if paddle
            else False
        )
        has_paddle_yaml = (
            paddle.get("yaml_entry", {}).get("found", False)
            if paddle
            else False
        )
        has_aliases = len(result.get("alias_candidates", [])) > 0

        if not has_pytorch:
            return "pytorch_not_found"

        if has_paddle_compat:
            return "verified_compat"

        if has_paddle_api:
            return "verified_api_h_only"

        if has_paddle_kernel:
            return "kernel_only"

        if has_paddle_yaml:
            return "yaml_only"

        if has_aliases:
            return "alias_candidate"

        return "truly_missing"

    def verify_batch(self, op_names: list, batch_name: str) -> dict:
        """Verify a batch of APIs."""
        results = []
        total = len(op_names)

        for idx, op_name in enumerate(op_names):
            print(f"  [{idx + 1}/{total}] Verifying {op_name}...")
            result = self.verify_op(op_name)
            results.append(result)

        return {
            "batch_name": batch_name,
            "total": total,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }


def generate_markdown_report(results: dict, output_path: str):
    """Generate human-readable Markdown report."""
    lines = []
    lines.append("# API 映射验证报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## 执行摘要")
    lines.append("")
    lines.append("| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |")
    lines.append("|------|------|--------|--------|----------|----------|")

    grand_total = 0
    grand_verified = 0
    grand_pending = 0
    grand_aliases = 0
    grand_missing = 0

    for batch_name, batch_result in results.items():
        total = batch_result.get("total", 0)
        verified = 0
        pending = 0
        aliases = 0
        missing = 0

        for r in batch_result.get("results", []):
            status = r.get("verification_status", "")
            if status in ("verified_compat", "verified_api_h_only"):
                verified += 1
            elif status in ("kernel_only", "yaml_only", "alias_candidate"):
                pending += 1
            if status == "alias_candidate":
                aliases += 1
            if status == "truly_missing":
                missing += 1

        lines.append(
            f"| {batch_name} | {total} | {verified} | {pending} | {aliases} | {missing} |"
        )
        grand_total += total
        grand_verified += verified
        grand_pending += pending
        grand_aliases += aliases
        grand_missing += missing

    lines.append(
        f"| **总计** | **{grand_total}** | **{grand_verified}** | **{grand_pending}** | **{grand_aliases}** | **{grand_missing}** |"
    )
    lines.append("")

    # Alias candidates detail
    all_alias_candidates = []
    for batch_result in results.values():
        for r in batch_result.get("results", []):
            if r.get("alias_candidates"):
                all_alias_candidates.extend(r["alias_candidates"])

    if all_alias_candidates:
        lines.append("## 发现的别名映射候选")
        lines.append("")
        lines.append("| PyTorch API | Paddle API | 规则 | 置信度 |")
        lines.append("|-------------|-----------|------|--------|")
        seen = set()
        for cand in all_alias_candidates:
            key = (cand["torch_api"], cand["paddle_api"])
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"| `at::{cand['torch_api']}` | `paddle::experimental::{cand['paddle_api']}` | {cand['rule']} | {cand['confidence']} |"
                )
        lines.append("")

    # Kernel-only candidates
    kernel_only = []
    for batch_result in results.values():
        for r in batch_result.get("results", []):
            if r.get("verification_status") == "kernel_only":
                paddle = r.get("paddle", {})
                kernel_info = paddle.get("kernel_registration", {})
                kernel_only.append(
                    {
                        "op": r["op_name"],
                        "cpu": kernel_info.get("cpu_registered", False),
                        "gpu": kernel_info.get("gpu_registered", False),
                    }
                )

    if kernel_only:
        lines.append("## Kernel 已注册但未暴露到 api.h 的候选")
        lines.append("")
        lines.append("| PyTorch API | CPU Kernel | GPU Kernel |")
        lines.append("|-------------|-----------|-----------|")
        for item in kernel_only:
            cpu_str = "是" if item["cpu"] else "否"
            gpu_str = "是" if item["gpu"] else "否"
            lines.append(f"| `at::{item['op']}` | {cpu_str} | {gpu_str} |")
        lines.append("")

    # Detailed results per batch
    lines.append("## 详细验证结果")
    lines.append("")

    for batch_name, batch_result in results.items():
        lines.append(f"### {batch_name}")
        lines.append("")

        status_counts = {}
        for r in batch_result.get("results", []):
            status = r.get("verification_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

        # List APIs with notes or non-standard status
        interesting = []
        for r in batch_result.get("results", []):
            status = r.get("verification_status", "")
            notes = r.get("notes", [])
            if status not in (
                "verified_compat",
                "verified_api_h_only",
                "truly_missing",
            ):
                interesting.append(r)
            elif notes:
                interesting.append(r)

        if interesting:
            lines.append("#### 需关注的 API")
            lines.append("")
            lines.append("| API | 状态 | 备注 |")
            lines.append("|-----|------|------|")
            for r in interesting[:50]:  # Limit to first 50
                op = r["op_name"]
                status = r.get("verification_status", "")
                notes = "; ".join(r.get("notes", []))[:100]
                lines.append(f"| `at::{op}` | {status} | {notes} |")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="基于 Step 2-1 方法论的 API 映射验证脚本"
    )
    parser.add_argument(
        "--mapping-file",
        default=DEFAULT_CONFIG["mapping_file"],
        help="现有映射表文件路径",
    )
    parser.add_argument(
        "--libtorch-ops-dir",
        default=DEFAULT_CONFIG["libtorch_ops_dir"],
        help="libtorch ops 头文件目录",
    )
    parser.add_argument(
        "--pytorch-src-dir",
        default=DEFAULT_CONFIG["pytorch_src_dir"],
        help="PyTorch 源码目录（用于追踪到 kernel 实现）",
    )
    parser.add_argument(
        "--paddle-src-dir",
        default=DEFAULT_CONFIG["paddle_src_dir"],
        help="Paddle 源码目录",
    )
    parser.add_argument(
        "--paddle-api-h",
        default=DEFAULT_CONFIG["paddle_api_h"],
        help="Paddle api.h 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "verification_output"),
        help="输出目录",
    )
    parser.add_argument(
        "--batch",
        default=None,
        help="仅处理指定批次（如 P0_exact_match, P4_missing）",
    )
    parser.add_argument(
        "--op",
        default=None,
        help="仅验证单个 API（如 abs）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="验证所有批次",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每批处理的最大数量",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load mapping
    print(f"Loading mapping from {args.mapping_file}...")
    mapping_data = parse_mapping_file(args.mapping_file)
    for cat, ops in mapping_data.items():
        print(f"  {cat}: {len(ops)} ops")

    # Create verifier
    config = {
        "libtorch_ops_dir": args.libtorch_ops_dir,
        "pytorch_src_dir": args.pytorch_src_dir,
        "paddle_src_dir": args.paddle_src_dir,
        "paddle_api_h": args.paddle_api_h,
    }
    verifier = APIVerifier(config)

    # Determine what to verify
    batches = create_batches(mapping_data)

    if args.op:
        # Single op mode
        print(f"\nVerifying single op: {args.op}")
        result = verifier.verify_op(args.op)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.batch:
        if args.batch not in batches:
            print(f"Unknown batch: {args.batch}")
            print(f"Available batches: {', '.join(batches.keys())}")
            return
        target_batches = {args.batch: batches[args.batch]}
    elif args.all:
        target_batches = batches
    else:
        # Default: verify P0 + P1 (highest priority)
        target_batches = {
            "P0_exact_match": batches.get("P0_exact_match", []),
            "P1_name_diff": batches.get("P1_name_diff", []),
        }

    # Execute verification
    all_results = {}
    for batch_name, op_names in target_batches.items():
        if args.limit:
            op_names = op_names[: args.limit]
        if not op_names:
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing batch: {batch_name} ({len(op_names)} ops)")
        print(f"{'=' * 60}")
        batch_result = verifier.verify_batch(op_names, batch_name)
        all_results[batch_name] = batch_result

    # Generate timestamped output filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_suffix = (
        f"_{args.batch}" if args.batch else ("_all" if args.all else "_default")
    )

    # Save JSON results
    json_path = os.path.join(
        args.output_dir, f"verification_results{batch_suffix}_{ts}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to: {json_path}")

    # Generate Markdown report
    report_path = os.path.join(
        args.output_dir, f"verification_report{batch_suffix}_{ts}.md"
    )
    generate_markdown_report(all_results, report_path)

    print("\nVerification complete!")


if __name__ == "__main__":
    import re

    main()
