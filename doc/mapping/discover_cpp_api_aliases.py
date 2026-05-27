#!/usr/bin/env python3
"""
自动发现 PyTorch C++ API (libtorch) 与 Paddle C++ API 之间的候选别名映射。

启发式规则：
1. 去掉 PyTorch 前缀 '_' 后的名称与 Paddle 函数名匹配
2. 命名风格翻转：conv_transposeNd ↔ convNd_transpose
3. 命名风格翻转：max_poolNd_with_indices ↔ max_poolNd_with_index
4. 已知功能映射：range → arange
5. 字符串相似度（Levenshtein ratio >= 0.85）作为补充发现

输出：cpp_api_alias_candidates.json，供 Agent 审核后生成 cpp_api_alias_mapping.json
"""

import argparse
import difflib
import json
import os
import re


def get_libtorch_ops(libtorch_ops_dir):
    files = [f for f in os.listdir(libtorch_ops_dir) if f.endswith(".h")]
    ops = set()
    skip = [
        "_dispatch",
        "_native",
        "_ops",
        "_composite",
        "_cpu",
        "_cuda",
        "_meta",
        "_backward",
        "_forward",
        "_functional",
        "_math",
        "_sparse",
        "_structured",
        "_quantized",
        "_mkldnn",
        "_vulkan",
        "_mps",
        "_xpu",
        "_hip",
        "_ort",
    ]
    for f in files:
        if any(s in f for s in skip):
            continue
        ops.add(os.path.splitext(f)[0])
    return ops


def get_paddle_funcs(paddle_api_h):
    with open(paddle_api_h, "r", encoding="utf-8") as fh:
        content = fh.read()
    return set(
        re.findall(
            r"PADDLE_API\s+(?:\w+::)?\w+(?:<[^>]+>)?\s+(\w+)\s*\(", content
        )
    )


def discover_aliases(libtorch_ops, paddle_funcs):
    candidates = []
    matched_torch = set()
    matched_paddle = set()

    # 规则1: 去掉前缀 '_' 匹配
    for op in libtorch_ops:
        if op.startswith("_"):
            stripped = op[1:]
            if stripped in paddle_funcs and stripped not in matched_paddle:
                candidates.append(
                    {
                        "torch_api": op,
                        "paddle_api": stripped,
                        "rule": "strip_underscore_prefix",
                        "confidence": "high",
                        "note": f"PyTorch 内部变体 `{op}` 对应标准实现 `{stripped}`",
                    }
                )
                matched_torch.add(op)
                matched_paddle.add(stripped)

    # 规则2: conv_transposeNd ↔ convNd_transpose
    for op in libtorch_ops:
        if op in matched_torch:
            continue
        m = re.match(r"conv_transpose(\d)d$", op)
        if m:
            nd = m.group(1)
            flipped = f"conv{nd}d_transpose"
            if flipped in paddle_funcs and flipped not in matched_paddle:
                candidates.append(
                    {
                        "torch_api": op,
                        "paddle_api": flipped,
                        "rule": "naming_style_flip_conv_transpose",
                        "confidence": "high",
                        "note": f"命名风格差异: `{op}` ↔ `{flipped}`",
                    }
                )
                matched_torch.add(op)
                matched_paddle.add(flipped)

    # 规则3: max_poolNd_with_indices ↔ max_poolNd_with_index
    for op in libtorch_ops:
        if op in matched_torch:
            continue
        m = re.match(r"max_pool(\d)d_with_indices$", op)
        if m:
            nd = m.group(1)
            flipped = f"max_pool{nd}d_with_index"
            if flipped in paddle_funcs and flipped not in matched_paddle:
                candidates.append(
                    {
                        "torch_api": op,
                        "paddle_api": flipped,
                        "rule": "naming_style_flip_indices",
                        "confidence": "high",
                        "note": f"命名风格差异: `{op}` ↔ `{flipped}`",
                    }
                )
                matched_torch.add(op)
                matched_paddle.add(flipped)

    # 规则4: 已知功能映射
    known_mappings = {
        "range": "arange",
    }
    for torch_op, paddle_op in known_mappings.items():
        if torch_op in libtorch_ops and paddle_op in paddle_funcs:
            if (
                torch_op not in matched_torch
                and paddle_op not in matched_paddle
            ):
                candidates.append(
                    {
                        "torch_api": torch_op,
                        "paddle_api": paddle_op,
                        "rule": "known_semantic_mapping",
                        "confidence": "high",
                        "note": f"已知功能映射: `{torch_op}` 在 Paddle 中实现为 `{paddle_op}`",
                    }
                )
                matched_torch.add(torch_op)
                matched_paddle.add(paddle_op)

    # 规则5: 字符串相似度补充（排除已匹配的，仅用于发现）
    remaining_torch = libtorch_ops - matched_torch
    remaining_paddle = paddle_funcs - matched_paddle
    for op in remaining_torch:
        matches = difflib.get_close_matches(
            op, remaining_paddle, n=3, cutoff=0.85
        )
        for m in matches:
            # 过滤掉明显不相关的
            if op == m:
                continue
            # 避免重复添加
            already = any(
                c["torch_api"] == op and c["paddle_api"] == m
                for c in candidates
            )
            if not already:
                candidates.append(
                    {
                        "torch_api": op,
                        "paddle_api": m,
                        "rule": "string_similarity",
                        "confidence": "medium",
                        "note": f"字符串相似度候选: `{op}` ↔ `{m}`，需 Agent 审核",
                    }
                )

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="自动发现 C++ API 候选别名映射"
    )

    # 优先从环境变量读取路径
    torch_dir = os.environ.get("TORCH_DIR", r"D:/Lenovo/libtorch")
    paddle_root = os.environ.get("PADDLE_ROOT", r"D:/Lenovo/Paddle")

    parser.add_argument(
        "--libtorch-ops-dir",
        default=os.path.join(torch_dir, "include", "ATen", "ops"),
        help="libtorch ATen/ops 头文件目录 (默认: $TORCH_DIR/include/ATen/ops)",
    )
    parser.add_argument(
        "--paddle-api-h",
        default=os.path.join(
            paddle_root, "paddle", "phi", "api", "include", "api.h"
        ),
        help="Paddle api.h 路径 (默认: $PADDLE_ROOT/paddle/phi/api/include/api.h)",
    )
    parser.add_argument(
        "--output",
        default="cpp_api_alias_candidates.json",
        help="输出候选别名 JSON 文件路径",
    )
    args = parser.parse_args()

    libtorch_ops = get_libtorch_ops(args.libtorch_ops_dir)
    paddle_funcs = get_paddle_funcs(args.paddle_api_h)

    print(f"Libtorch ops: {len(libtorch_ops)}")
    print(f"Paddle funcs: {len(paddle_funcs)}")

    candidates = discover_aliases(libtorch_ops, paddle_funcs)

    # 按规则排序
    rule_order = {
        "strip_underscore_prefix": 0,
        "naming_style_flip_conv_transpose": 1,
        "naming_style_flip_indices": 2,
        "known_semantic_mapping": 3,
        "string_similarity": 4,
    }
    candidates.sort(
        key=lambda x: (rule_order.get(x["rule"], 99), x["torch_api"])
    )

    high_conf = [c for c in candidates if c["confidence"] == "high"]
    medium_conf = [c for c in candidates if c["confidence"] == "medium"]

    print(f"\n高置信度候选: {len(high_conf)}")
    for c in high_conf:
        print(f"  {c['torch_api']} -> {c['paddle_api']} ({c['rule']})")

    print(f"\n中置信度候选: {len(medium_conf)}")
    for c in medium_conf[:20]:
        print(f"  {c['torch_api']} -> {c['paddle_api']} ({c['rule']})")
    if len(medium_conf) > 20:
        print(f"  ... 还有 {len(medium_conf) - 20} 个")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    print(f"\n已保存候选列表到: {args.output}")


if __name__ == "__main__":
    main()
