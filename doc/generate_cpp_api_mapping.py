#!/usr/bin/env python3
"""
自动生成 PyTorch C++ API (libtorch) 与 Paddle C++ API 映射表。

分类逻辑：
1. API 完全一致：Paddle compat 层已有对应头文件。
2. 功能缺失：libtorch 中有但 Paddle（compat + api.h）均没有。
3. 其他差异：compat 层没有但 api.h 中有同名实现。
   对这部分函数，直接解析 C++ 头文件中的函数签名，对比参数列表
   （参数名、类型、默认值、数量）进行自动分类。
"""

import argparse
import json
import os
import re

# ============================================================
# C++ Signature Parser
# ============================================================


def parse_libtorch_signature(filepath):
    """
    从 libtorch ops 头文件中提取主签名。
    排除 _out / _outf 变体，在剩余重载中取参数最少的一个作为主签名。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 定位 namespace at { ... } 区块
    ns_match = re.search(
        r"namespace\s+at\s*\{([\s\S]*?)\}\s*\n*(#else|#endif|$)", content
    )
    if not ns_match:
        return None
    ns_body = ns_match.group(1)

    # 匹配 inline [ret] func_name(params) {
    pattern = r"inline\s+([\w:&\s\*<>,]+?)\s+(\w+)\s*\((.*?)\)\s*\{"
    matches = re.findall(pattern, ns_body, re.DOTALL)

    candidates = []
    for ret, name, params in matches:
        # 跳过 _out / _outf 变体
        if name.endswith("_out") or name.endswith("_outf"):
            continue
        params = params.strip()
        arg_list = _split_args(params)
        candidates.append(
            {
                "ret": ret.strip(),
                "name": name,
                "params": params,
                "arg_count": len(arg_list),
                "args": arg_list,
            }
        )

    if not candidates:
        return None

    # 取参数最少的作为主签名（通常是用户最常用的版本）
    candidates.sort(key=lambda x: x["arg_count"])
    return candidates[0]


def parse_paddle_signatures(api_h_path):
    """从 Paddle api.h 中提取所有函数签名，同函数名保留参数最少的一个。"""
    with open(api_h_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配 PADDLE_API [ret] func_name(params);
    pattern = r"PADDLE_API\s+([\w:&\s\*<>,]+?)\s+(\w+)\s*\((.*?)\);"
    matches = re.findall(pattern, content, re.DOTALL)

    sigs = {}
    for ret, name, params in matches:
        params = params.strip()
        arg_list = _split_args(params)
        entry = {
            "ret": ret.strip(),
            "name": name,
            "params": params,
            "arg_count": len(arg_list),
            "args": arg_list,
        }
        if name not in sigs or sigs[name]["arg_count"] > entry["arg_count"]:
            sigs[name] = entry

    return sigs


def _split_args(params_str):
    """按逗号分割参数，尊重嵌套括号。"""
    if not params_str:
        return []
    args = []
    depth = 0
    current = []
    for ch in params_str:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(ch)
    arg = "".join(current).strip()
    if arg:
        args.append(arg)
    return args


# ============================================================
# Argument Parsing & Normalization
# ============================================================


def parse_arg(arg_str):
    """将单个参数字符串解析为 dict: {type, name, default}"""
    arg_str = arg_str.strip()
    if not arg_str:
        return None

    default = None
    if "=" in arg_str:
        idx = arg_str.index("=")
        default = arg_str[idx + 1 :].strip()
        arg_str = arg_str[:idx].strip()

    tokens = arg_str.split()
    if len(tokens) >= 2:
        name = tokens[-1]
        type_str = " ".join(tokens[:-1])
        name = name.lstrip("*").lstrip("&")
    else:
        name = arg_str
        type_str = ""

    return {"type": type_str, "name": name, "default": default}


def normalize_type(t):
    """归一化 C++ 类型名，用于跨框架对比。"""
    t = t.strip()
    # 去掉开头的 const/volatile
    t = re.sub(r"^\s*(const|volatile)\s+", "", t)
    # 去掉结尾的 & / *
    t = re.sub(r"\s*[&\*]\s*$", "", t).strip()
    # 常见别名统一
    replacements = [
        ("at::Tensor", "Tensor"),
        ("at::Scalar", "Scalar"),
        ("c10::ScalarType", "ScalarType"),
        ("std::vector<int64_t>", "IntArray"),
        ("std::optional", "optional"),
        ("paddle::optional", "optional"),
        ("std::string", "string"),
        ("std::tuple", "tuple"),
        ("c10::Device", "Device"),
        ("c10::Layout", "Layout"),
        ("c10::MemoryFormat", "MemoryFormat"),
        ("at::IntArrayRef", "IntArray"),
        ("IntArrayRef", "IntArray"),
        ("::std::optional", "optional"),
        ("::std::tuple", "tuple"),
        ("::std::vector", "std::vector"),
    ]
    for old, new in replacements:
        t = t.replace(old, new)
    # 压缩多余空格
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_cpp_api_alias_mapping(mapping_path=None):
    """
    加载 C++ API 别名映射文件。
    格式: { torch_api_name: paddle_api_name }
    """
    if mapping_path is None:
        mapping_path = os.path.join(
            os.path.dirname(__file__), "cpp_api_alias_mapping.json"
        )
    if not os.path.exists(mapping_path):
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_default_value(val):
    """将分数形式的默认值（如 '1.0f/8'）规范化为十进制字符串（如 '0.125f'）。"""
    if not isinstance(val, str):
        return val
    import re

    m = re.match(r"^(\d+\.?\d*)f?/\d+\.?\d*$", val)
    if m:
        try:
            result = eval(val.replace("f", ""))
            # 保留 f 后缀
            suffix = "f" if "f" in val else ""
            if isinstance(result, float):
                return f"{result:.16g}{suffix}"
        except Exception:
            pass
    return val


# 常见参数名别名映射（PyTorch -> Paddle）
ARG_NAME_ALIASES = {
    "self": ["x", "input", "values"],
    "other": ["y"],
    "dim": ["axis"],
    "dims": ["axis"],
    "input": ["x"],
    "weight": ["filter", "scale"],
    "src": ["x"],
    "tensor": ["x"],
    "indices": ["x"],
    "target": ["label"],
    "source": ["add_value"],
    "eps": ["epsilon"],
    "row": ["rows"],
    "col": ["cols"],
    "pad": ["paddings"],
    "value": ["pad_value"],
    "training": ["is_test"],
}


def _append_default_diff(remark, t_arg, p_arg):
    """若两参数默认值不同，在 remark 后追加默认值差异说明。"""
    t_def_raw = t_arg.get("default")
    p_def_raw = p_arg.get("default")
    t_def = normalize_default_value(t_def_raw)
    p_def = normalize_default_value(p_def_raw)
    if t_def == p_def:
        return remark
    # 尝试数值比较，避免 1e-05 vs 1e-5 这类等价但字符串不同的假阳性
    if t_def is not None and p_def is not None:
        try:
            t_num = float(str(t_def).replace("f", ""))
            p_num = float(str(p_def).replace("f", ""))
            if t_num == p_num:
                return remark
        except (ValueError, TypeError):
            pass
    t_name = t_arg["name"]
    p_name = p_arg["name"]
    if t_def is not None and p_def is not None:
        remark += f" 默认值不同：PyTorch 默认 `{t_name}={t_def}`，Paddle 默认 `{p_name}={p_def}`。"
    elif t_def is not None:
        remark += (
            f" 默认值不同：PyTorch 默认 `{t_name}={t_def}`，Paddle 无默认值。"
        )
    elif p_def is not None:
        remark += (
            f" 默认值不同：PyTorch 无默认值，Paddle 默认 `{p_name}={p_def}`。"
        )
    return remark


def _args_match_by_name(t_name, p_name):
    """判断两个参数名是否匹配（含别名映射）。"""
    if t_name == p_name:
        return True
    aliases = ARG_NAME_ALIASES.get(t_name, [])
    return p_name in aliases


# ============================================================
# Signature Comparison Engine
# ============================================================


def compare_signatures(torch_sig, paddle_sig):
    """
    对比两个函数签名，返回 (category, detail)。
    分类优先级（从高到低）：
      1. 返回类型不一致
      2. 参数数量不一致 → torch 参数更多 / paddle 参数更多
      3. 参数类型不一致
      4. 参数默认值不一致
      5. 仅参数名不一致
      6. 仅 API 调用方式不一致（兜底）
    """
    if not torch_sig or not paddle_sig:
        return ("仅 API 调用方式不一致", "签名解析失败")

    t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
    p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]

    # 排除 Paddle 内部特有的 predefined_out 参数
    p_args = [a for a in p_args if "predefined_out" not in a["name"]]

    t_ret = normalize_type(torch_sig["ret"])
    p_ret = normalize_type(paddle_sig["ret"])

    # 1. 返回类型
    if t_ret != p_ret and t_ret and p_ret:
        return (
            "返回参数类型不一致",
            "返回类型不一致",
        )

    # 2. 按参数名（含别名）做跨索引匹配，计算实际未匹配数量
    t_matched = set()
    p_matched = set()
    for i, t_arg in enumerate(t_args):
        for j, p_arg in enumerate(p_args):
            if j in p_matched:
                continue
            if _args_match_by_name(t_arg["name"], p_arg["name"]):
                t_matched.add(i)
                p_matched.add(j)
                break

    unmatched_t = len(t_args) - len(t_matched)
    unmatched_p = len(p_args) - len(p_matched)

    if unmatched_p > unmatched_t:
        return (
            "paddle 参数更多",
            f"PyTorch {len(t_args)} 个参数，Paddle {len(p_args)} 个参数（跨名匹配后 Paddle 多 {unmatched_p - unmatched_t} 个）",
        )
    if unmatched_t > unmatched_p:
        return (
            "torch 参数更多",
            f"PyTorch {len(t_args)} 个参数，Paddle {len(p_args)} 个参数（跨名匹配后 PyTorch 多 {unmatched_t - unmatched_p} 个）",
        )

    # 3-5. 对已成功匹配的参数逐对对比类型/默认值/名称
    type_diff = False
    default_diff = False
    name_diff = False

    for i, t_arg in enumerate(t_args):
        if i not in t_matched:
            continue
        # 找到匹配的 Paddle 参数索引
        matched_j = None
        for j, p_arg in enumerate(p_args):
            if j in p_matched and _args_match_by_name(
                t_arg["name"], p_arg["name"]
            ):
                matched_j = j
                break
        if matched_j is None:
            continue

        t_type = normalize_type(t_arg["type"])
        p_type = normalize_type(p_args[matched_j]["type"])

        if t_type != p_type:
            type_diff = True

        t_def = t_arg["default"]
        p_def = p_args[matched_j]["default"]
        if t_def != p_def:
            default_diff = True

        if t_arg["name"] != p_args[matched_j]["name"]:
            name_diff = True

    if type_diff:
        return ("输入参数类型不一致", "存在参数类型差异")

    if default_diff:
        return ("参数默认值不一致", "存在参数默认值差异")

    if name_diff:
        return (
            "仅参数名不一致",
            "参数类型和默认值相同，仅参数名不同",
        )

    return (
        "仅 API 调用方式不一致",
        "签名高度相似，调用方式或语义有细微差异",
    )


# ============================================================
# Main Generator
# ============================================================


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


def get_compat_ops(paddle_compat_dir):
    return {
        os.path.splitext(f)[0]
        for f in os.listdir(paddle_compat_dir)
        if f.endswith(".h")
    }


def get_api_h_funcs(paddle_api_h):
    with open(paddle_api_h, "r", encoding="utf-8") as fh:
        content = fh.read()
    return set(
        re.findall(
            r"PADDLE_API\s+(?:\w+::)?\w+(?:<[^>]+>)?\s+(\w+)\s*\(", content
        )
    )


def generate_mapping(
    libtorch_ops_dir,
    paddle_compat_dir,
    paddle_api_h,
):
    libtorch_ops = get_libtorch_ops(libtorch_ops_dir)
    compat_ops = get_compat_ops(paddle_compat_dir)
    api_funcs = get_api_h_funcs(paddle_api_h)

    # 加载别名映射
    alias_map = load_cpp_api_alias_mapping()

    # 基于别名映射，找出 Paddle 中有实现但名称不同的 PyTorch ops
    aliased_torch_ops = set()
    for torch_name, paddle_name in alias_map.items():
        if torch_name in libtorch_ops and paddle_name in api_funcs:
            aliased_torch_ops.add(torch_name)

    exact_match = sorted(compat_ops & libtorch_ops)
    invoke_diff_ops = sorted((api_funcs & libtorch_ops) - compat_ops)
    # 将别名匹配的 ops 也纳入差异对比
    invoke_diff_ops += sorted(aliased_torch_ops - compat_ops)
    missing = sorted(libtorch_ops - api_funcs - compat_ops - aliased_torch_ops)

    # 分类到差异文档子目录的映射（用于生成可跳转链接）
    CATEGORY_TO_DOC_DIR = {
        "仅参数名不一致": "cpp_args_name_diff",
        "参数默认值不一致": "cpp_args_default_value_diff",
        "输入参数类型不一致": "cpp_input_args_type_diff",
        "返回参数类型不一致": "cpp_output_args_type_diff",
        "paddle 参数更多": "cpp_paddle_more_args",
        "torch 参数更多": "cpp_torch_more_args",
        "API 别名": "cpp_api_alias_diff",
        "语义差异": "cpp_semantic_mismatch",
    }

    # 预解析 Paddle 签名
    paddle_sigs = parse_paddle_signatures(paddle_api_h)

    # 对 invoke_diff 做签名级自动分类
    invoke_categories = {}

    # 语义差异 API（PyTorch 与 Paddle 同名但语义不同）
    SEMANTIC_MISMATCH_APIS = {"uniform"}

    for op in invoke_diff_ops:
        torch_path = os.path.join(libtorch_ops_dir, f"{op}.h")
        torch_sig = parse_libtorch_signature(torch_path)
        # 通过别名映射获取正确的 Paddle API 名称
        paddle_op_name = alias_map.get(op, op)
        paddle_sig = paddle_sigs.get(paddle_op_name)

        # namespace 为空或解析不到签名的函数不纳入映射表
        if not torch_sig or not paddle_sig:
            continue

        # 语义差异 API 直接归入特殊分类
        if op in SEMANTIC_MISMATCH_APIS:
            cat = "语义差异"
            detail = "PyTorch 与 Paddle API 语义不同，不应视为等价实现"
        else:
            cat, detail = compare_signatures(torch_sig, paddle_sig)

        # 如果该 op 是通过别名映射识别的，同时加入 "API 别名" 分类
        if op in alias_map:
            if "API 别名" not in invoke_categories:
                invoke_categories["API 别名"] = []
            invoke_categories["API 别名"].append(
                (op, detail, torch_sig, paddle_sig)
            )

        if cat not in invoke_categories:
            invoke_categories[cat] = []
        invoke_categories[cat].append((op, detail, torch_sig, paddle_sig))

    # 分类固定顺序
    section_order = [
        "仅 API 调用方式不一致",
        "仅参数名不一致",
        "paddle 参数更多",
        "参数默认值不一致",
        "torch 参数更多",
        "输入参数用法不一致",
        "输入参数类型不一致",
        "返回参数类型不一致",
        "组合替代实现",
        "API 别名",
        "语义差异",
    ]

    # 收集统计
    stats = {}

    lines = []
    lines.append("# PyTorch C++ API (libtorch) 与 Paddle C++ API 映射表")
    lines.append("")
    lines.append(
        "本文梳理了 PyTorch C++ API (libtorch) 与 PaddlePaddle C++ API 的对应关系与差异分析，"
    )
    lines.append("帮助开发者快速迁移 PyTorch C++ 使用经验。")
    lines.append("")
    lines.append(
        "> **Note**: 本映射表基于以下路径**自动解析 C++ 函数签名**生成："
    )
    lines.append(f"> - PyTorch C++ API 头文件: `{libtorch_ops_dir}`")
    lines.append(f"> - Paddle compat 层头文件: `{paddle_compat_dir}`")
    lines.append(f"> - Paddle `api.h` 头文件: `{paddle_api_h}`")
    lines.append("")
    lines.append(
        "> **说明**: 对于 compat 层未封装的函数，脚本直接对比 libtorch 头文件与 `paddle::experimental` "
        "命名空间中同名函数的**返回类型、参数类型、参数名、参数默认值、参数数量**，按优先级自动归入差异分类。"
    )
    lines.append("")

    # 映射分类总表
    lines.append("## API 映射分类")
    lines.append("")
    lines.append("| 序号 | 类别 | 简介 |")
    lines.append("| ---- | ---- | ---- |")
    lines.append(
        "| 1 | API 完全一致 | compat 层已实现与 PyTorch C++ API 完全一致的接口，可直接替换命名空间使用 |"
    )
    lines.append(
        "| 2 | 仅 API 调用方式不一致 | Paddle 有同名实现，但调用方式与 PyTorch 不一致（签名解析后兜底分类） |"
    )
    lines.append("| 3 | 仅参数名不一致 | 功能相同，但部分参数名称不同 |")
    lines.append("| 4 | paddle 参数更多 | Paddle 中提供了更多可选参数 |")
    lines.append("| 5 | 参数默认值不一致 | 功能相同，但某些参数的默认值不同 |")
    lines.append("| 6 | torch 参数更多 | PyTorch 中提供了更多参数 |")
    lines.append("| 7 | 输入参数用法不一致 | 对输入参数的处理方式不同 |")
    lines.append("| 8 | 输入参数类型不一致 | 要求的输入数据类型不同 |")
    lines.append("| 9 | 返回参数类型不一致 | 返回值的类型或结构不同 |")
    lines.append(
        "| 10 | 组合替代实现 | 在 Paddle 中没有直接对应的单一 API，需要多个 API 组合实现 |"
    )
    lines.append(
        "| 11 | API 别名 | PyTorch 与 Paddle 功能一致，但 API 名称不同 |"
    )
    lines.append(
        "| 12 | 功能缺失 | PyTorch C++ API 的功能在 Paddle 中暂时没有等效实现 |"
    )
    lines.append("")

    # 1. API 完全一致
    cat_name = "API 完全一致"
    stats[cat_name] = len(exact_match)
    lines.append(f"### 1. {cat_name}")
    lines.append("")
    lines.append(
        "**简介：** compat 层已实现与 PyTorch C++ API 完全一致的接口，只需将代码中的命名空间或调用方式按 compat 层声明使用即可。"
    )
    lines.append("")
    lines.append(
        "| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |"
    )
    lines.append(
        "|------|-----------------|----------------|----------|------|"
    )
    for idx, op in enumerate(exact_match, 1):
        lines.append(
            f"| {idx} | `at::{op}` | `at::{op}` (compat层) | {cat_name} | - |"
        )
    lines.append("")

    # 2-11. 差异分类
    for section_idx, cat_name in enumerate(section_order, 2):
        ops = invoke_categories.get(cat_name, [])
        stats[cat_name] = len(ops)

        lines.append(f"### {section_idx}. {cat_name}")
        lines.append("")

        if cat_name == "仅 API 调用方式不一致":
            lines.append(
                "**简介：** Paddle `paddle::experimental` 命名空间中有同名实现，但 compat 层尚未提供完全一致的封装。"
                "以下函数经签名对比后归入此类，多为调用语义或底层实现差异。"
            )
        elif cat_name == "仅参数名不一致":
            lines.append("**简介：** 此类 API 功能相同，但部分参数名称不同。")
        elif cat_name == "paddle 参数更多":
            lines.append("**简介：** 此类 API 在 Paddle 中提供了更多可选参数。")
        elif cat_name == "参数默认值不一致":
            lines.append(
                "**简介：** 此类 API 功能相同，但某些参数的默认值不同。"
            )
        elif cat_name == "torch 参数更多":
            lines.append("**简介：** 此类 API 在 PyTorch 中提供了更多参数。")
        elif cat_name == "输入参数用法不一致":
            lines.append("**简介：** 此类 API 对输入参数的处理方式不同。")
        elif cat_name == "输入参数类型不一致":
            lines.append("**简介：** 此类 API 要求的输入数据类型不同。")
        elif cat_name == "返回参数类型不一致":
            lines.append("**简介：** 此类 API 返回值的类型或结构不同。")
        elif cat_name == "组合替代实现":
            lines.append(
                "**简介：** 此类功能在 Paddle 中没有直接对应的单一 API，需要通过多个 Paddle API 组合来实现。"
            )
        elif cat_name == "API 别名":
            lines.append(
                "**简介：** 此类 PyTorch API 在 Paddle 中有功能一致的实现，但 API 名称不同。"
            )

        lines.append("")
        lines.append(
            "| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |"
        )
        lines.append(
            "|------|-----------------|----------------|----------|------|"
        )
        for idx, (op, detail, torch_sig, paddle_sig) in enumerate(ops, 1):
            doc_dir = CATEGORY_TO_DOC_DIR.get(cat_name)
            if doc_dir:
                remark = f"[差异对比]({doc_dir}/at.{op}.md)"
            else:
                # 仅 API 调用方式不一致 / 组合替代实现（无差异文档）
                remark = detail
            if cat_name == "API 别名":
                paddle_op_name = alias_map.get(op, op)
                lines.append(
                    f"| {idx} | `at::{op}` | `paddle::experimental::{paddle_op_name}` | {cat_name} | {remark} |"
                )
            else:
                lines.append(
                    f"| {idx} | `at::{op}` | `paddle::experimental::{op}` | {cat_name} | {remark} |"
                )
        if not ops:
            lines.append("| - | - | - | - | 暂无 |")
        lines.append("")

    # 13. 功能缺失
    cat_name = "功能缺失"
    stats[cat_name] = len(missing)
    lines.append(f"### 13. {cat_name}")
    lines.append("")
    lines.append(
        "**简介：** 此类 PyTorch C++ API 在 Paddle 中暂时没有等效实现。"
    )
    lines.append("")
    lines.append(
        "| 序号 | PyTorch C++ API | Paddle C++ API | 映射分类 | 备注 |"
    )
    lines.append(
        "|------|-----------------|----------------|----------|------|"
    )
    for idx, op in enumerate(missing, 1):
        lines.append(f"| {idx} | `at::{op}` | - | {cat_name} | - |")
    lines.append("")

    # 统计
    lines.append("## 统计")
    lines.append("")
    total_classified = sum(stats.get(c, 0) for c in section_order)
    lines.append(f"- **API 完全一致**: {stats['API 完全一致']} 个")
    for cat_name in section_order:
        lines.append(f"- **{cat_name}**: {stats.get(cat_name, 0)} 个")
    lines.append(f"- **功能缺失**: {stats['功能缺失']} 个")
    lines.append(f"- **API 别名映射数**: {len(alias_map)} 个")
    total_parsed = (
        len(exact_match)
        + sum(len(v) for v in invoke_categories.values())
        + len(missing)
    )
    lines.append(f"- **libtorch 主 ops 总数**: {len(libtorch_ops)} 个")
    lines.append(f"- **实际参与映射的 ops 数**: {total_parsed} 个")
    lines.append("")

    return "\n".join(lines), invoke_categories


def generate_cpp_args_name_diff_docs(invoke_categories, output_dir):
    """为'仅参数名不一致'分类的函数生成独立的 C++ API 参数名差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    name_diff_ops = invoke_categories.get("仅参数名不一致", [])
    generated = []
    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in name_diff_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        # 排除 predefined_out
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        # 构建简化签名（仅参数名+默认值）
        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        # 构建参数映射表格
        diff_rows = []
        max_len = max(len(t_args), len(p_args))
        for i in range(max_len):
            t_name = t_args[i]["name"] if i < len(t_args) else "-"
            p_name = p_args[i]["name"] if i < len(p_args) else "-"
            if t_name != p_name:
                remark = f"仅参数名不一致，`{t_name}` 对应 `{p_name}`。"
            else:
                remark = "参数名一致。"
            diff_rows.append(f"| {t_name} | {p_name} | {remark} |")

        lines = []
        lines.append(f"## [仅参数名不一致]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        lines.append("两者功能一致且参数用法一致，仅参数名不一致，具体如下：")
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(f"成功生成 {len(generated)} 个 C++ 参数名差异文档到: {output_dir}")
    return generated


def generate_cpp_paddle_more_args_docs(invoke_categories, output_dir):
    """为'paddle 参数更多'分类的函数生成独立的 C++ API 参数数量差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    paddle_more_ops = invoke_categories.get("paddle 参数更多", [])
    generated = []

    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in paddle_more_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        # 按参数名匹配，避免索引错位
        p_dict = {a["name"]: a for a in p_args}
        matched_p = set()
        diff_rows = []

        for t_arg in t_args:
            t_name = t_arg["name"]
            if t_name in p_dict:
                remark = "参数名一致。"
                if op == "embedding" and t_name == "weight":
                    remark = "参数名一致，但位置顺序不同：PyTorch 第 1 个参数 weight 对应 Paddle 第 2 个参数 weight，调用时需按名传参或调换位置。"
                remark = _append_default_diff(remark, t_arg, p_dict[t_name])
                diff_rows.append(f"| {t_name} | {t_name} | {remark} |")
                matched_p.add(t_name)
            elif t_name in ARG_NAME_ALIASES:
                matched_alias = None
                for alias in ARG_NAME_ALIASES[t_name]:
                    if alias in p_dict:
                        matched_alias = alias
                        break
                if matched_alias:
                    remark = (
                        f"仅参数名不一致，`{t_name}` 对应 `{matched_alias}`。"
                    )
                    remark = _append_default_diff(
                        remark, t_arg, p_dict[matched_alias]
                    )
                    diff_rows.append(
                        f"| {t_name} | {matched_alias} | {remark} |"
                    )
                    matched_p.add(matched_alias)
                else:
                    remark = f"Paddle 无此参数，PyTorch 有 `{t_name}`。"
                    if op in ("add", "index_add") and t_name == "alpha":
                        remark = "影响计算语义，PyTorch 计算 self + alpha * other，Paddle 无此参数，等价表达需组合调用。"
                    elif op == "subtract" and t_name == "alpha":
                        remark = "影响计算语义，PyTorch 计算 self - alpha * other，Paddle 无此参数，等价表达需组合调用。"
                    diff_rows.append(f"| {t_name} | - | {remark} |")
            else:
                remark = f"Paddle 无此参数，PyTorch 有 `{t_name}`。"
                if op in ("add", "index_add") and t_name == "alpha":
                    remark = "影响计算语义，PyTorch 计算 self + alpha * other，Paddle 无此参数，等价表达需组合调用。"
                elif op == "subtract" and t_name == "alpha":
                    remark = "影响计算语义，PyTorch 计算 self - alpha * other，Paddle 无此参数，等价表达需组合调用。"
                diff_rows.append(f"| {t_name} | - | {remark} |")

        for p_arg in p_args:
            p_name = p_arg["name"]
            if p_name not in matched_p:
                diff_rows.append(
                    f"| - | {p_name} | PyTorch 无此参数，Paddle 有 `{p_name}`。 |"
                )

        lines = []
        lines.append(f"## [paddle 参数更多]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        if op == "var":
            lines.append(
                "两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下："
            )
            lines.append("")
            lines.append(
                "> 注：当前对比基于 PyTorch 最简重载 `var(self, unbiased)`。PyTorch 也存在带 `dim/keepdim/correction` 的完整重载。"
            )
        else:
            lines.append(
                "两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下："
            )
        lines.append("")
        lines.append("> 注：参数映射表按 PyTorch 签名顺序排列。")
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(
        f"成功生成 {len(generated)} 个 C++ paddle 参数更多差异文档到: {output_dir}"
    )
    return generated


def generate_cpp_torch_more_args_docs(invoke_categories, output_dir):
    """为'torch 参数更多'分类的函数生成独立的 C++ API 参数数量差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    torch_more_ops = invoke_categories.get("torch 参数更多", [])
    generated = []

    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in torch_more_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        # 按参数名匹配，避免索引错位
        p_dict = {a["name"]: a for a in p_args}
        matched_p = set()
        diff_rows = []

        for t_arg in t_args:
            t_name = t_arg["name"]
            if t_name in p_dict:
                remark = "参数名一致。"
                if op == "embedding" and t_name == "weight":
                    remark = "参数名一致，但位置顺序不同：PyTorch 第 1 个参数 weight 对应 Paddle 第 2 个参数 weight，调用时需按名传参或调换位置。"
                remark = _append_default_diff(remark, t_arg, p_dict[t_name])
                diff_rows.append(f"| {t_name} | {t_name} | {remark} |")
                matched_p.add(t_name)
            elif t_name in ARG_NAME_ALIASES:
                matched_alias = None
                for alias in ARG_NAME_ALIASES[t_name]:
                    if alias in p_dict:
                        matched_alias = alias
                        break
                if matched_alias:
                    remark = (
                        f"仅参数名不一致，`{t_name}` 对应 `{matched_alias}`。"
                    )
                    if (
                        op == "rrelu"
                        and t_name == "training"
                        and matched_alias == "is_test"
                    ):
                        remark = "【需对值取反】`training` 与 `is_test` 语义互为反义，不能直接搬运布尔值（training=true 时应设 is_test=false）。"
                    elif op in ("add", "index_add") and t_name == "alpha":
                        remark = "影响计算语义，PyTorch 计算 self + alpha * other，Paddle 无此参数，等价表达需组合调用。"
                    elif op == "subtract" and t_name == "alpha":
                        remark = "影响计算语义，PyTorch 计算 self - alpha * other，Paddle 无此参数，等价表达需组合调用。"
                    remark = _append_default_diff(
                        remark, t_arg, p_dict[matched_alias]
                    )
                    diff_rows.append(
                        f"| {t_name} | {matched_alias} | {remark} |"
                    )
                    matched_p.add(matched_alias)
                else:
                    diff_rows.append(
                        f"| {t_name} | - | Paddle 无此参数，PyTorch 有 `{t_name}`。 |"
                    )
            else:
                remark = f"Paddle 无此参数，PyTorch 有 `{t_name}`。"
                if op in ("add", "index_add") and t_name == "alpha":
                    remark = "影响计算语义，PyTorch 计算 self + alpha * other，Paddle 无此参数，等价表达需组合调用。"
                elif op == "subtract" and t_name == "alpha":
                    remark = "影响计算语义，PyTorch 计算 self - alpha * other，Paddle 无此参数，等价表达需组合调用。"
                diff_rows.append(f"| {t_name} | - | {remark} |")

        for p_arg in p_args:
            p_name = p_arg["name"]
            if p_name not in matched_p:
                diff_rows.append(
                    f"| - | {p_name} | PyTorch 无此参数，Paddle 有 `{p_name}`。 |"
                )

        lines = []
        lines.append(f"## [torch 参数更多]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        lines.append("PyTorch 相比 Paddle 支持更多参数，具体如下：")
        lines.append("")
        lines.append("> 注：参数映射表按 PyTorch 签名顺序排列。")
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(
        f"成功生成 {len(generated)} 个 C++ torch 参数更多差异文档到: {output_dir}"
    )
    return generated


def generate_cpp_api_alias_diff_docs(invoke_categories, output_dir):
    """为'API 别名'分类的函数生成独立的 C++ API 别名差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    alias_ops = invoke_categories.get("API 别名", [])
    generated = []
    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in alias_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        paddle_op_name = alias_map.get(op, op)

        # 按参数名匹配
        p_dict = {a["name"]: a for a in p_args}
        matched_p = set()
        diff_rows = []

        for t_arg in t_args:
            t_name = t_arg["name"]
            if t_name in p_dict:
                diff_rows.append(f"| {t_name} | {t_name} | 参数名一致。 |")
                matched_p.add(t_name)
            elif t_name in ARG_NAME_ALIASES:
                matched_alias = None
                for alias in ARG_NAME_ALIASES[t_name]:
                    if alias in p_dict:
                        matched_alias = alias
                        break
                if matched_alias:
                    diff_rows.append(
                        f"| {t_name} | {matched_alias} | 仅参数名不一致，`{t_name}` 对应 `{matched_alias}`。 |"
                    )
                    matched_p.add(matched_alias)
                else:
                    diff_rows.append(
                        f"| {t_name} | - | Paddle 无此参数，PyTorch 有 `{t_name}`。 |"
                    )
            else:
                diff_rows.append(
                    f"| {t_name} | - | Paddle 无此参数，PyTorch 有 `{t_name}`。 |"
                )

        for p_arg in p_args:
            p_name = p_arg["name"]
            if p_name not in matched_p:
                diff_rows.append(
                    f"| - | {p_name} | PyTorch 无此参数，Paddle 有 `{p_name}`。 |"
                )

        lines = []
        lines.append(f"## [API 别名]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        lines.append(
            f"两者功能一致，但 API 名称不同，PyTorch 为 `{op}`，Paddle 为 `{paddle_op_name}`。参数映射具体如下："
        )
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(f"成功生成 {len(generated)} 个 C++ API 别名差异文档到: {output_dir}")
    return generated


def generate_cpp_semantic_mismatch_docs(invoke_categories, output_dir):
    """为'语义差异'分类的函数生成独立的 C++ API 语义差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    mismatch_ops = invoke_categories.get("语义差异", [])
    generated = []
    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in mismatch_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        # 按参数名匹配
        p_dict = {a["name"]: a for a in p_args}
        matched_p = set()
        diff_rows = []

        for t_arg in t_args:
            t_name = t_arg["name"]
            if t_name in p_dict:
                diff_rows.append(f"| {t_name} | {t_name} | 参数名一致。 |")
                matched_p.add(t_name)
            elif t_name in ARG_NAME_ALIASES:
                matched_alias = None
                for alias in ARG_NAME_ALIASES[t_name]:
                    if alias in p_dict:
                        matched_alias = alias
                        break
                if matched_alias:
                    diff_rows.append(
                        f"| {t_name} | {matched_alias} | 仅参数名不一致，`{t_name}` 对应 `{matched_alias}`。 |"
                    )
                    matched_p.add(matched_alias)
                else:
                    diff_rows.append(
                        f"| {t_name} | - | Paddle 无此参数，PyTorch 有 `{t_name}`。 |"
                    )
            else:
                diff_rows.append(
                    f"| {t_name} | - | Paddle 无此参数，PyTorch 有 `{t_name}`。 |"
                )

        for p_arg in p_args:
            p_name = p_arg["name"]
            if p_name not in matched_p:
                diff_rows.append(
                    f"| - | {p_name} | PyTorch 无此参数，Paddle 有 `{p_name}`。 |"
                )

        lines = []
        lines.append(f"## [语义差异]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        lines.append(
            f"**注意：PyTorch `at::{op}` 与 Paddle `paddle::experimental::{paddle_op_name}` 语义不同，不应视为等价 API。**"
        )
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(f"成功生成 {len(generated)} 个 C++ API 语义差异文档到: {output_dir}")
    return generated


def generate_cpp_output_args_type_diff_docs(invoke_categories, output_dir):
    """为'返回参数类型不一致'分类的函数生成独立的 C++ API 返回类型差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    ret_diff_ops = invoke_categories.get("返回参数类型不一致", [])
    generated = []

    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in ret_diff_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        t_ret = normalize_type(torch_sig["ret"])
        p_ret = normalize_type(paddle_sig["ret"])

        lines = []
        lines.append(f"## [返回参数类型不一致]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str}) -> {t_ret}")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str}) -> {p_ret}"
        )
        lines.append("```")
        lines.append("")
        lines.append("两者功能一致，但返回类型不一致，具体如下：")
        lines.append("")
        lines.append("### 返回类型映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.append(f"| {t_ret} | {p_ret} | 返回类型不一致。 |")
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(
        f"成功生成 {len(generated)} 个 C++ 返回参数类型差异文档到: {output_dir}"
    )
    return generated


def generate_cpp_input_args_type_diff_docs(invoke_categories, output_dir):
    """为'输入参数类型不一致'分类的函数生成独立的 C++ API 参数类型差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    type_diff_ops = invoke_categories.get("输入参数类型不一致", [])
    generated = []
    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in type_diff_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        diff_rows = []
        max_len = max(len(t_args), len(p_args))
        for i in range(max_len):
            t_name = t_args[i]["name"] if i < len(t_args) else "-"
            p_name = p_args[i]["name"] if i < len(p_args) else "-"
            t_type = (
                normalize_type(t_args[i]["type"]) if i < len(t_args) else "-"
            )
            p_type = (
                normalize_type(p_args[i]["type"]) if i < len(p_args) else "-"
            )

            remark = ""
            if t_type != p_type and t_name == p_name:
                remark = f"参数类型不一致，PyTorch 为 `{t_type}`，Paddle 为 `{p_type}`。"
            elif t_type != p_type and t_name != p_name:
                remark = f"参数名与类型均不一致，PyTorch `{t_name}` (`{t_type}`) 对应 Paddle `{p_name}` (`{p_type}`)。"
            elif t_name != p_name:
                remark = f"仅参数名不一致，`{t_name}` 对应 `{p_name}`。"
            else:
                remark = "参数名与类型均一致。"
            diff_rows.append(f"| {t_name} | {p_name} | {remark} |")

        lines = []
        lines.append(f"## [输入参数类型不一致]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        lines.append("两者功能一致，但输入参数类型不一致，具体如下：")
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(
        f"成功生成 {len(generated)} 个 C++ 输入参数类型差异文档到: {output_dir}"
    )
    return generated


def generate_cpp_args_default_value_diff_docs(invoke_categories, output_dir):
    """为'参数默认值不一致'分类的函数生成独立的 C++ API 参数默认值差异文档。"""
    os.makedirs(output_dir, exist_ok=True)

    default_diff_ops = invoke_categories.get("参数默认值不一致", [])
    generated = []
    alias_map = load_cpp_api_alias_mapping()

    for op, detail, torch_sig, paddle_sig in default_diff_ops:
        t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
        p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]
        p_args = [a for a in p_args if "predefined_out" not in a["name"]]

        paddle_op_name = alias_map.get(op, op)

        def fmt_sig(args):
            parts = []
            for a in args:
                name = a["name"]
                if a["default"] is not None:
                    parts.append(
                        f"{name}={normalize_default_value(a['default'])}"
                    )
                else:
                    parts.append(name)
            return ", ".join(parts)

        torch_sig_str = fmt_sig(t_args)
        paddle_sig_str = fmt_sig(p_args)

        diff_rows = []
        max_len = max(len(t_args), len(p_args))
        for i in range(max_len):
            t_name = t_args[i]["name"] if i < len(t_args) else "-"
            p_name = p_args[i]["name"] if i < len(p_args) else "-"
            t_def = t_args[i]["default"] if i < len(t_args) else None
            p_def = p_args[i]["default"] if i < len(p_args) else None

            remark = ""
            if t_def != p_def:
                t_def_str = f"`{t_def}`" if t_def is not None else "无默认值"
                p_def_str = f"`{p_def}`" if p_def is not None else "无默认值"
                remark = f"参数默认值不一致，PyTorch 为 {t_def_str}，Paddle 为 {p_def_str}。"
            elif t_name != p_name:
                remark = f"仅参数名不一致，`{t_name}` 对应 `{p_name}`。"
            else:
                remark = "参数名与默认值均一致。"
            diff_rows.append(f"| {t_name} | {p_name} | {remark} |")

        lines = []
        lines.append(f"## [参数默认值不一致]at::{op}")
        lines.append("")
        lines.append("### PyTorch C++ API")
        lines.append("```cpp")
        lines.append(f"at::{op}({torch_sig_str})")
        lines.append("```")
        lines.append("")
        lines.append("### Paddle C++ API")
        lines.append("```cpp")
        lines.append(
            f"paddle::experimental::{paddle_op_name}({paddle_sig_str})"
        )
        lines.append("```")
        lines.append("")
        lines.append("两者功能一致，但参数默认值不一致，具体如下：")
        lines.append("")
        lines.append("### 参数映射")
        lines.append("")
        lines.append("| PyTorch C++ | Paddle C++ | 备注 |")
        lines.append("| ----------- | ---------- | ---- |")
        lines.extend(diff_rows)
        lines.append("")

        filepath = os.path.join(output_dir, f"at.{op}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        generated.append(op)

    print(
        f"成功生成 {len(generated)} 个 C++ 参数默认值差异文档到: {output_dir}"
    )
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="自动生成 PyTorch C++ API 与 Paddle C++ API 映射表"
    )
    parser.add_argument(
        "--libtorch-ops-dir",
        default=r"D:/Lenovo/libtorch/include/ATen/ops",
        help="libtorch ATen/ops 头文件目录",
    )
    parser.add_argument(
        "--paddle-compat-dir",
        default=r"D:/Lenovo/Paddle/paddle/phi/api/include/compat/ATen/ops",
        help="Paddle compat 层头文件目录",
    )
    parser.add_argument(
        "--paddle-api-h",
        default=r"D:/Lenovo/Paddle/paddle/phi/api/include/api.h",
        help="Paddle api.h 头文件路径",
    )
    parser.add_argument(
        "--output",
        default="cpp_api_mapping_cn.md",
        help="输出 Markdown 文件路径",
    )
    args = parser.parse_args()

    content, invoke_categories = generate_mapping(
        args.libtorch_ops_dir,
        args.paddle_compat_dir,
        args.paddle_api_h,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"成功生成映射表: {args.output}")

    # 同时生成 C++ API 参数名差异文档
    cpp_args_dir = os.path.join(
        os.path.dirname(args.output), "cpp_args_name_diff"
    )
    generate_cpp_args_name_diff_docs(invoke_categories, cpp_args_dir)

    # 同时生成 C++ API 参数默认值差异文档
    cpp_default_dir = os.path.join(
        os.path.dirname(args.output), "cpp_args_default_value_diff"
    )
    generate_cpp_args_default_value_diff_docs(
        invoke_categories, cpp_default_dir
    )

    # 同时生成 C++ API 输入参数类型差异文档
    cpp_type_dir = os.path.join(
        os.path.dirname(args.output), "cpp_input_args_type_diff"
    )
    generate_cpp_input_args_type_diff_docs(invoke_categories, cpp_type_dir)

    # 同时生成 C++ API 返回参数类型差异文档
    cpp_output_dir = os.path.join(
        os.path.dirname(args.output), "cpp_output_args_type_diff"
    )
    generate_cpp_output_args_type_diff_docs(invoke_categories, cpp_output_dir)

    # 同时生成 C++ API paddle 参数更多差异文档
    cpp_paddle_more_dir = os.path.join(
        os.path.dirname(args.output), "cpp_paddle_more_args"
    )
    generate_cpp_paddle_more_args_docs(invoke_categories, cpp_paddle_more_dir)

    # 同时生成 C++ API torch 参数更多差异文档
    cpp_torch_more_dir = os.path.join(
        os.path.dirname(args.output), "cpp_torch_more_args"
    )
    generate_cpp_torch_more_args_docs(invoke_categories, cpp_torch_more_dir)

    # 同时生成 C++ API 别名差异文档
    cpp_alias_dir = os.path.join(
        os.path.dirname(args.output), "cpp_api_alias_diff"
    )
    generate_cpp_api_alias_diff_docs(invoke_categories, cpp_alias_dir)

    # 同时生成 C++ API 语义差异文档
    cpp_semantic_dir = os.path.join(
        os.path.dirname(args.output), "cpp_semantic_mismatch"
    )
    generate_cpp_semantic_mismatch_docs(invoke_categories, cpp_semantic_dir)


if __name__ == "__main__":
    main()
