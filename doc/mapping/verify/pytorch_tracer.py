"""
PyTorch C++ API implementation trace module.
Follows Step 2-1 methodology: libtorch header -> native_functions.yaml -> kernel impl.
"""

import os
import re


class PyTorchTracer:
    """Trace PyTorch C++ API implementation chain following Step 2-1 methodology."""

    def __init__(
        self, libtorch_ops_dir: str, pytorch_src_dir: str | None = None
    ):
        self.libtorch_ops_dir = libtorch_ops_dir
        self.pytorch_src_dir = pytorch_src_dir

    def trace(self, op_name: str) -> dict:
        """
        Trace a single API's full implementation chain.

        Returns structured trace record:
        {
            "op_name": str,
            "declaration": {...},     # header file analysis
            "dispatcher": {...},      # ops struct analysis
            "yaml_entry": {...},      # native_functions.yaml entry
            "kernel_impl": {...},     # kernel implementation location
            "notes": [str]
        }
        """
        result = {"op_name": op_name, "notes": []}

        # Step 1: Parse libtorch header declaration
        result["declaration"] = self._parse_header(op_name)

        # Step 2: Parse ops.h dispatcher struct
        result["dispatcher"] = self._parse_ops_struct(op_name)

        # Step 3: Parse native_functions.yaml (needs pytorch_src_dir)
        if self.pytorch_src_dir:
            result["yaml_entry"] = self._parse_native_functions_yaml(op_name)
            # Step 4: Locate kernel implementation
            if result["yaml_entry"].get("found"):
                result["kernel_impl"] = self._locate_kernel(
                    op_name, result["yaml_entry"]
                )

        return result

    def _parse_header(self, op_name: str) -> dict:
        """Parse libtorch ops header file for declaration info."""
        header_path = os.path.join(self.libtorch_ops_dir, f"{op_name}.h")
        if not os.path.exists(header_path):
            return {"found": False}

        with open(header_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract schema strings
        schemas = re.findall(
            r'static constexpr const char\* schema_str = "([^"]+)"', content
        )

        # Extract function signatures: inline [ret] name(params) {
        sigs = re.findall(
            r"inline\s+([\w:&\s\*<>,]+?)\s+(\w+)\s*\((.*?)\)\s*\{",
            content,
            re.DOTALL,
        )

        signatures = []
        for ret, name, params in sigs:
            if name.endswith("_out") or name.endswith("_outf"):
                continue
            signatures.append(
                {
                    "ret": ret.strip(),
                    "name": name,
                    "params": params.strip(),
                }
            )

        # Check if it's dispatcher forwarded (contains at::_ops::<op>::call)
        is_dispatcher = "at::_ops::" in content and "::call(" in content

        return {
            "found": True,
            "header_file": f"ATen/ops/{op_name}.h",
            "schemas": schemas,
            "signatures": signatures,
            "is_dispatcher_forwarded": is_dispatcher,
        }

    def _parse_ops_struct(self, op_name: str) -> dict:
        """Parse generated ops struct header for dispatcher info."""
        ops_header = os.path.join(self.libtorch_ops_dir, f"{op_name}_ops.h")
        if not os.path.exists(ops_header):
            return {"found": False}

        with open(ops_header, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract TORCH_API struct definitions
        structs = []
        struct_pattern = re.compile(
            r"struct\s+(?:TORCH_API\s+)?(\w+)\s*\{[^}]*?"
            r'static\s+constexpr\s+const\s+char\*\s+name\s*=\s*"([^"]+)";'
            r'[^}]*?static\s+constexpr\s+const\s+char\*\s+overload_name\s*=\s*"([^"]*)";'
            r'[^}]*?static\s+constexpr\s+const\s+char\*\s+schema_str\s*=\s*"([^"]+)";',
            re.DOTALL,
        )

        for match in struct_pattern.finditer(content):
            structs.append(
                {
                    "struct_name": match.group(1),
                    "aten_name": match.group(2),
                    "overload_name": match.group(3),
                    "schema": match.group(4),
                }
            )

        # Also check for redispatch / call patterns
        has_call = "::call(" in content
        has_redispatch = "::redispatch(" in content

        return {
            "found": True,
            "structs": structs,
            "has_call": has_call,
            "has_redispatch": has_redispatch,
        }

    def _parse_native_functions_yaml(self, op_name: str) -> dict:
        """Find schema and dispatch info in native_functions.yaml."""
        yaml_path = os.path.join(
            self.pytorch_src_dir, "aten/src/ATen/native/native_functions.yaml"
        )
        if not os.path.exists(yaml_path):
            return {"found": False, "note": "native_functions.yaml not found"}

        entries = self._find_yaml_entries(yaml_path, op_name)
        return {"found": len(entries) > 0, "entries": entries}

    def _find_yaml_entries(self, yaml_path: str, op_name: str) -> list:
        """Find all YAML entries for a given op name."""
        entries = []
        current_entry = None
        in_entry = False
        indent_level = 0

        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # New entry starts
            if stripped.startswith("- func: "):
                if current_entry:
                    entries.append(current_entry)

                func_def = stripped[len("- func: ") :].strip()
                # Extract op name: op_name.overload(...) or op_name(...)
                entry_op = func_def.split(".")[0].split("(")[0].strip()

                if entry_op == op_name:
                    current_entry = {
                        "func": func_def,
                        "dispatch": {},
                        "structured": False,
                        "tags": [],
                        "variants": None,
                    }
                    in_entry = True
                    indent_level = len(line) - len(line.lstrip())
                else:
                    current_entry = None
                    in_entry = False

            elif in_entry and current_entry:
                # Check if we've exited this entry (new top-level key or empty line with less indent)
                if stripped.startswith("- func: ") or (
                    stripped
                    and not stripped.startswith("#")
                    and len(line) - len(line.lstrip()) <= indent_level
                ):
                    entries.append(current_entry)
                    current_entry = None
                    in_entry = False
                    i -= 1  # Re-process this line
                    continue

                # Parse dispatch section
                if stripped == "dispatch:":
                    i += 1
                    while i < len(lines):
                        dispatch_line = lines[i]
                        dispatch_stripped = dispatch_line.strip()
                        if not dispatch_stripped:
                            i += 1
                            continue
                        # Exit dispatch section on de-indent
                        dl_indent = len(dispatch_line) - len(
                            dispatch_line.lstrip()
                        )
                        if dl_indent <= len(line) - len(line.lstrip()):
                            i -= 1
                            break
                        if ":" in dispatch_stripped:
                            parts = dispatch_stripped.split(":", 1)
                            key = parts[0].strip()
                            val = parts[1].strip()
                            if key and val:
                                current_entry["dispatch"][key] = val
                        i += 1

                elif stripped.startswith("structured:"):
                    current_entry["structured"] = "True" in stripped

                elif stripped.startswith("structured_inherits:"):
                    parts = stripped.split(":", 1)
                    if len(parts) > 1:
                        current_entry["structured_inherits"] = parts[1].strip()

                elif stripped.startswith("variants:"):
                    parts = stripped.split(":", 1)
                    if len(parts) > 1:
                        current_entry["variants"] = parts[1].strip()

                elif stripped.startswith("tags:"):
                    tag_match = re.search(r"tags:\s*\[([^\]]*)\]", stripped)
                    if tag_match:
                        current_entry["tags"] = [
                            t.strip()
                            for t in tag_match.group(1).split(",")
                            if t.strip()
                        ]

            i += 1

        if current_entry:
            entries.append(current_entry)

        return entries

    def _locate_kernel(self, op_name: str, yaml_entry: dict) -> dict:
        """Locate kernel implementation files from YAML dispatch info."""
        result = {}

        for entry in yaml_entry.get("entries", []):
            for backend, kernel_func in entry.get("dispatch", {}).items():
                # Skip composite dispatch entries - they point back to the op itself
                if backend.startswith("Composite") and kernel_func == op_name:
                    result["composite"] = True
                    result["composite_backend"] = backend
                    continue

                file_path = self._find_kernel_file(kernel_func, backend)
                if file_path:
                    key = f"{backend.lower()}_file"
                    result[key] = file_path
                    result[f"{backend.lower()}_func"] = kernel_func

        return result

    def _find_kernel_file(self, kernel_func: str, backend: str) -> str:
        """Search PyTorch source for kernel function definition."""
        search_dirs = {
            "CPU": ["aten/src/ATen/native/"],
            "CUDA": ["aten/src/ATen/native/cuda/"],
            "MPS": ["aten/src/ATen/native/mps/"],
            "Meta": ["aten/src/ATen/native/"],
        }

        # Handle composite backends
        if backend.startswith("Composite"):
            search_dirs["Composite"] = ["aten/src/ATen/native/"]

        for search_dir in search_dirs.get(backend, ["aten/src/ATen/native/"]):
            full_dir = os.path.join(self.pytorch_src_dir, search_dir)
            if not os.path.exists(full_dir):
                continue

            for root, _, files in os.walk(full_dir):
                for file in files:
                    if file.endswith((".cpp", ".cu", ".h")):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                content = f.read()
                            # Match function definitions: [ret] func_name(
                            # Support Tensor&, Tensor, void, etc.
                            pattern = (
                                r"(?:Tensor|Tensor\&|void|Scalar|bool|int64_t|"
                                + rf"at::Tensor|const Tensor)\s+{re.escape(kernel_func)}\s*\("
                            )
                            if re.search(pattern, content):
                                return os.path.relpath(
                                    filepath, self.pytorch_src_dir
                                )
                        except (OSError, UnicodeDecodeError):
                            continue
        return None
