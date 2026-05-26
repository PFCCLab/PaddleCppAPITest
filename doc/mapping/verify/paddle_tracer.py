"""
Paddle C++ API implementation trace module.
Traces: api.h declaration -> ops.yaml config -> kernel registration -> compat layer.
"""

import os
import re


class PaddleTracer:
    """Trace Paddle C++ API implementation chain."""

    def __init__(self, paddle_api_h: str, paddle_src_dir: str):
        self.paddle_api_h = paddle_api_h
        self.paddle_src_dir = paddle_src_dir
        self._api_h_content = None
        self._api_h_funcs = None

    def _get_api_h_content(self) -> str:
        if self._api_h_content is None:
            if os.path.exists(self.paddle_api_h):
                with open(self.paddle_api_h, "r", encoding="utf-8") as f:
                    self._api_h_content = f.read()
            else:
                self._api_h_content = ""
        return self._api_h_content

    def trace(self, op_name: str) -> dict:
        """
        Trace a single Paddle API's implementation chain.

        Returns:
        {
            "op_name": str,
            "api_declaration": {...},
            "yaml_entry": {...},
            "kernel_registration": {...},
            "compat_layer": {...},
            "notes": [str]
        }
        """
        result = {"op_name": op_name, "notes": []}

        # Step 1: Check api.h
        result["api_declaration"] = self._check_api_h(op_name)

        # Step 2: Parse ops.yaml
        result["yaml_entry"] = self._parse_ops_yaml(op_name)

        # Step 3: Check kernel registration
        result["kernel_registration"] = self._check_kernel_registration(op_name)

        # Step 4: Check compat layer
        result["compat_layer"] = self._check_compat_layer(op_name)

        return result

    def _check_api_h(self, op_name: str) -> dict:
        """Check if function exists in api.h."""
        content = self._get_api_h_content()
        if not content:
            return {"found": False, "note": "api.h not found"}

        # Match function declaration: PADDLE_API ... op_name(
        # Be careful with overloads and template types
        pattern = rf"PADDLE_API\s+(?:[\w:]+\s+)*?{re.escape(op_name)}\s*\("
        match = re.search(pattern, content)

        if not match:
            return {"found": False}

        # Extract full signature (find from match start to semicolon)
        start = match.start()
        semicolon_pos = content.find(";", start)
        if semicolon_pos == -1:
            return {"found": False, "note": "signature incomplete"}

        signature = content[start : semicolon_pos + 1]

        return {
            "found": True,
            "signature": signature,
            "has_predefined_out": "predefined_out" in signature,
        }

    def _parse_ops_yaml(self, op_name: str) -> dict:
        """Find op config in Paddle ops YAML files."""
        yaml_paths = [
            os.path.join(self.paddle_src_dir, "paddle/phi/ops/yaml/ops.yaml"),
            os.path.join(
                self.paddle_src_dir, "paddle/phi/ops/yaml/sparse_ops.yaml"
            ),
            os.path.join(
                self.paddle_src_dir, "paddle/phi/ops/yaml/fused_ops.yaml"
            ),
            os.path.join(
                self.paddle_src_dir,
                "paddle/phi/ops/yaml/inconsistent/static_ops.yaml",
            ),
        ]

        for yaml_path in yaml_paths:
            if not os.path.exists(yaml_path):
                continue

            entry = self._find_yaml_entry(yaml_path, op_name)
            if entry:
                entry["file"] = os.path.relpath(yaml_path, self.paddle_src_dir)
                return entry

        return {"found": False}

    def _find_yaml_entry(self, yaml_path: str, op_name: str) -> dict:
        """Find a specific op entry in a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Match op entry: - op : op_name
        pattern = rf"^- op\s*:\s*{re.escape(op_name)}\s*\n((?:  .*\n)*)"
        match = re.search(pattern, content, re.MULTILINE)

        if not match:
            return None

        entry_text = match.group(1)
        result = {"found": True}

        # Extract args
        args_match = re.search(r"args\s*:\s*\(([^)]*)\)", entry_text)
        if args_match:
            result["args"] = args_match.group(1).strip()

        # Extract output
        output_match = re.search(r"output\s*:\s*(.+?)(?:\n|$)", entry_text)
        if output_match:
            result["output"] = output_match.group(1).strip()

        # Extract kernel func
        kernel_match = re.search(
            r"kernel\s*:\s*\n((?:    .*\n|\n)*)", entry_text
        )
        if kernel_match:
            kernel_text = kernel_match.group(1)
            func_match = re.search(r"func\s*:\s*(\w+)", kernel_text)
            if func_match:
                result["kernel"] = {"func": func_match.group(1)}
            data_type_match = re.search(r"data_type\s*:\s*(\w+)", kernel_text)
            if data_type_match:
                if "kernel" not in result:
                    result["kernel"] = {}
                result["kernel"]["data_type"] = data_type_match.group(1)

        # Extract inplace
        inplace_match = re.search(r"inplace\s*:\s*\(([^)]+)\)", entry_text)
        if inplace_match:
            result["inplace"] = inplace_match.group(1).strip()

        # Extract backward
        backward_match = re.search(r"backward\s*:\s*(\w+)", entry_text)
        if backward_match:
            result["backward"] = backward_match.group(1).strip()

        # Extract infer_meta
        infer_meta_match = re.search(
            r"infer_meta\s*:\s*\n((?:    .*\n|\n)*)", entry_text
        )
        if infer_meta_match:
            infer_text = infer_meta_match.group(1)
            func_match = re.search(r"func\s*:\s*(\w+)", infer_text)
            if func_match:
                result["infer_meta"] = {"func": func_match.group(1)}

        return result

    def _check_kernel_registration(self, op_name: str) -> dict:
        """Check if kernel is registered for CPU/GPU."""
        result = {
            "found": False,
            "cpu_registered": False,
            "gpu_registered": False,
        }

        kernels_dir = os.path.join(self.paddle_src_dir, "paddle/phi/kernels")
        if not os.path.exists(kernels_dir):
            return result

        # Search CPU kernel registration
        cpu_pattern = (
            rf"PD_REGISTER_KERNEL\s*\(\s*{re.escape(op_name)}\s*,\s*CPU"
        )
        cpu_dir = os.path.join(kernels_dir, "cpu")
        if os.path.exists(cpu_dir):
            result["cpu_registered"] = self._search_dir_for_pattern(
                cpu_dir, cpu_pattern, ".cc"
            )

        # Search GPU kernel registration
        gpu_pattern = (
            rf"PD_REGISTER_KERNEL\s*\(\s*{re.escape(op_name)}\s*,\s*(GPU|CUDA)"
        )
        gpu_dir = os.path.join(kernels_dir, "gpu")
        if os.path.exists(gpu_dir):
            result["gpu_registered"] = self._search_dir_for_pattern(
                gpu_dir, gpu_pattern, (".cu", ".cc")
            )

        # Search for activation macro registrations
        if not result["cpu_registered"]:
            macro_patterns = [
                rf"PD_REGISTER_ACTIVATION_KERNEL\s*\(\s*{re.escape(op_name)}\s*,",
                rf"PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX\s*\(\s*{re.escape(op_name)}\s*,",
            ]
            for macro_pat in macro_patterns:
                if self._search_dir_for_pattern(kernels_dir, macro_pat, ".cc"):
                    result["cpu_registered"] = True
                    break

        # Check for kernel header declaration
        header_path = os.path.join(kernels_dir, f"{op_name}_kernel.h")
        if os.path.exists(header_path):
            result["has_header"] = True
            result["header_file"] = os.path.relpath(
                header_path, self.paddle_src_dir
            )

        result["found"] = result["cpu_registered"] or result["gpu_registered"]
        return result

    def _search_dir_for_pattern(
        self, directory: str, pattern: str, extensions
    ) -> bool:
        """Search directory for files matching pattern."""
        if isinstance(extensions, str):
            extensions = (extensions,)

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extensions):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            if re.search(pattern, f.read()):
                                return True
                    except (OSError, UnicodeDecodeError):
                        continue
        return False

    def _check_compat_layer(self, op_name: str) -> dict:
        """Check if compat layer already has a wrapper."""
        compat_dir = os.path.join(
            self.paddle_src_dir,
            "paddle/phi/api/include/compat/ATen/ops",
        )
        compat_file = os.path.join(compat_dir, f"{op_name}.h")

        if os.path.exists(compat_file):
            return {
                "found": True,
                "file": f"compat/ATen/ops/{op_name}.h",
            }

        return {"found": False}

    def get_all_api_h_funcs(self) -> set:
        """Return all function names found in api.h."""
        if self._api_h_funcs is not None:
            return self._api_h_funcs

        content = self._get_api_h_content()
        pattern = r"PADDLE_API\s+(?:[\w:]+\s+)*?(\w+)\s*\("
        self._api_h_funcs = set(re.findall(pattern, content))
        return self._api_h_funcs
