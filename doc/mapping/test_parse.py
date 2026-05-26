import os
import re


def parse_libtorch_signature(filepath):
    """从 libtorch ops 头文件中提取主签名（排除 out/outf 变体，取参数最少）"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find namespace at { ... }
    ns_match = re.search(
        r"namespace\s+at\s*\{([\s\S]*?)\}\s*\n*(#else|#endif|$)", content
    )
    if not ns_match:
        return None
    ns_body = ns_match.group(1)

    # Pattern: inline [ret] func_name(params) {
    pattern = r"inline\s+([\w:&\s\*<>]+?)\s+(\w+)\s*\((.*?)\)\s*\{"
    matches = re.findall(pattern, ns_body, re.DOTALL)

    candidates = []
    for ret, name, params in matches:
        # Skip _out and _outf variants
        if name.endswith("_out") or name.endswith("_outf"):
            continue
        params = params.strip()
        arg_list = []
        if params:
            depth = 0
            current = []
            for ch in params:
                if ch in "([{<":
                    depth += 1
                elif ch in ")]}>":
                    depth -= 1
                elif ch == "," and depth == 0:
                    arg = "".join(current).strip()
                    if arg:
                        arg_list.append(arg)
                    current = []
                    continue
                current.append(ch)
            arg = "".join(current).strip()
            if arg:
                arg_list.append(arg)

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

    candidates.sort(key=lambda x: x["arg_count"])
    return candidates[0]


def parse_paddle_signatures(api_h_path):
    """从 api.h 中提取所有函数签名"""
    with open(api_h_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern: PADDLE_API [ret] func_name(params);
    pattern = r"PADDLE_API\s+([\w:&\s\*<>,]+?)\s+(\w+)\s*\((.*?)\);"
    matches = re.findall(pattern, content, re.DOTALL)

    sigs = {}
    for ret, name, params in matches:
        params = params.strip()
        arg_list = []
        if params:
            depth = 0
            current = []
            for ch in params:
                if ch in "([{<":
                    depth += 1
                elif ch in ")]}>":
                    depth -= 1
                elif ch == "," and depth == 0:
                    arg = "".join(current).strip()
                    if arg:
                        arg_list.append(arg)
                    current = []
                    continue
                current.append(ch)
            arg = "".join(current).strip()
            if arg:
                arg_list.append(arg)

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


def parse_arg(arg_str):
    """Parse a single argument into (type, name, default)"""
    arg_str = arg_str.strip()
    if not arg_str:
        return None

    default = None
    if "=" in arg_str:
        idx = arg_str.index("=")
        default = arg_str[idx + 1 :].strip()
        arg_str = arg_str[:idx].strip()

    # The name is typically the last token
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
    """Normalize C++ type names for comparison"""
    t = t.strip()
    t = re.sub(r"^\s*(const|volatile)\s+", "", t)
    t = re.sub(r"\s*[&\*]\s*$", "", t).strip()
    t = t.replace("at::Tensor", "Tensor")
    t = t.replace("at::Scalar", "Scalar")
    t = t.replace("c10::ScalarType", "ScalarType")
    t = t.replace("std::vector<int64_t>", "IntArray")
    t = t.replace("std::optional", "optional")
    t = t.replace("paddle::optional", "optional")
    t = t.replace("std::string", "string")
    t = t.replace("std::tuple", "tuple")
    t = t.replace("c10::Device", "Device")
    t = t.replace("c10::Layout", "Layout")
    t = t.replace("c10::MemoryFormat", "MemoryFormat")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compare_signatures(torch_sig, paddle_sig):
    """
    Compare two signatures and return a classification.
    Returns (category, details)
    """
    if not torch_sig or not paddle_sig:
        return ("仅 API 调用方式不一致", "无法解析签名")

    t_args = [parse_arg(a) for a in torch_sig["args"] if parse_arg(a)]
    p_args = [parse_arg(a) for a in paddle_sig["args"] if parse_arg(a)]

    # Exclude paddle predefined_out parameter
    p_args = [a for a in p_args if "predefined_out" not in a["name"]]

    t_ret = normalize_type(torch_sig["ret"])
    p_ret = normalize_type(paddle_sig["ret"])

    # Check return type
    if t_ret != p_ret and t_ret and p_ret:
        return (
            "返回参数类型不一致",
            f"PyTorch返回: {t_ret}, Paddle返回: {p_ret}",
        )

    # Check parameter count
    if len(p_args) > len(t_args):
        return (
            "paddle 参数更多",
            f"PyTorch: {len(t_args)}个, Paddle: {len(p_args)}个",
        )
    if len(t_args) > len(p_args):
        return (
            "torch 参数更多",
            f"PyTorch: {len(t_args)}个, Paddle: {len(p_args)}个",
        )

    # Same parameter count, compare each
    type_diff = False
    default_diff = False
    name_diff = False

    for i in range(len(t_args)):
        t_type = normalize_type(t_args[i]["type"])
        p_type = normalize_type(p_args[i]["type"])

        if t_type != p_type:
            type_diff = True

        if t_args[i]["default"] != p_args[i]["default"]:
            default_diff = True

        if t_args[i]["name"] != p_args[i]["name"]:
            name_diff = True

    if type_diff:
        return ("输入参数类型不一致", "存在参数类型差异")

    if default_diff:
        return ("参数默认值不一致", "存在参数默认值差异")

    if name_diff:
        return ("仅参数名不一致", "参数类型和默认值相同，仅参数名不同")

    return ("仅 API 调用方式不一致", "签名高度相似，调用方式或语义有细微差异")


if __name__ == "__main__":
    libtorch_dir = "D:/迅雷下载/libtorch/include/ATen/ops"
    paddle_api_h = "D:/Lenovo/Paddle/paddle/phi/api/include/api.h"

    test_ops = [
        "acos",
        "add",
        "atan2",
        "relu",
        "batch_norm",
        "conv2d",
        "rrelu",
        "asin",
        "celu",
        "dropout",
        "elu",
        "exp",
    ]

    paddle_sigs = parse_paddle_signatures(paddle_api_h)

    for op in test_ops:
        torch_path = os.path.join(libtorch_dir, f"{op}.h")
        if not os.path.exists(torch_path):
            print(f"{op}: libtorch header not found")
            continue

        torch_sig = parse_libtorch_signature(torch_path)
        paddle_sig = paddle_sigs.get(op)

        if not torch_sig:
            print(f"{op}: failed to parse torch signature")
            continue
        if not paddle_sig:
            print(f"{op}: failed to parse paddle signature")
            continue

        cat, detail = compare_signatures(torch_sig, paddle_sig)
        print(f"{op}: {cat} ({detail})")
        print(f"  torch: {torch_sig['ret']} {op}({torch_sig['params']})")
        print(f"  paddle: {paddle_sig['ret']} {op}({paddle_sig['params']})")
        print()
