#!/usr/bin/env python3
"""
Fix mapping table based on verification results.
"""

import re


def parse_mapping_sections(content: str) -> dict:
    """Parse mapping markdown into sections with table rows."""
    sections = {}
    current_section = None
    current_rows = []
    header_lines = []
    in_table = False
    header_idx = 0
    separator_idx = 0

    lines = content.split("\n")

    for i, line in enumerate(lines):
        # Detect section headers
        section_match = re.match(r"### (\d+)\.\s*(.+)", line)
        if section_match:
            if current_section and current_rows:
                sections[current_section] = {
                    "rows": current_rows,
                    "header_idx": header_idx,
                    "separator_idx": separator_idx,
                }
            current_section = section_match.group(2).strip()
            current_rows = []
            in_table = False
            continue

        # Detect table header
        if (
            current_section
            and line.startswith("| 序号 ")
            and "PyTorch C++ API" in line
        ):
            header_lines = [line]
            in_table = True
            header_idx = i
            continue

        # Detect table separator
        if in_table and line.startswith("|---"):
            header_lines.append(line)
            separator_idx = i
            continue

        # Detect table data rows
        if in_table and line.startswith("|") and "at::" in line:
            current_rows.append(line)
            continue

        # End of table
        if in_table and not line.startswith("|") and line.strip():
            in_table = False

    # Save last section
    if current_section and current_rows:
        sections[current_section] = {
            "rows": current_rows,
            "header_idx": header_idx,
            "separator_idx": separator_idx,
        }

    return sections


def extract_api_name(row: str) -> str:
    """Extract API name from a table row."""
    match = re.search(r"`at::(\w+)`", row)
    return match.group(1) if match else None


def renumber_rows(rows: list) -> list:
    """Renumber rows starting from 1."""
    new_rows = []
    for idx, row in enumerate(rows, 1):
        # Replace the first number in the row (sequence number)
        parts = row.split("|")
        if len(parts) >= 2:
            # parts[0] is empty (before first |), parts[1] is the sequence number
            parts[1] = f" {idx} "
            new_row = "|".join(parts)
            new_rows.append(new_row)
    return new_rows


def main():
    with open("cpp_api_mapping_cn.md", "r", encoding="utf-8") as f:
        content = f.read()

    sections = parse_mapping_sections(content)

    # Define fixes
    # Format: {api_name: {'remove_from': [section_names], 'add_to': section_name or None}}
    fixes = {
        "_conj": {"remove_from": ["仅参数名不一致"], "add_to": None},
        "log_sigmoid": {"remove_from": ["仅参数名不一致"], "add_to": None},
        "_aminmax": {
            "remove_from": ["返回参数类型不一致", "API 别名"],
            "add_to": "功能缺失",
        },
        "_unique": {
            "remove_from": ["返回参数类型不一致", "API 别名"],
            "add_to": "功能缺失",
        },
        "max_pool2d_with_indices": {
            "remove_from": ["返回参数类型不一致", "API 别名"],
            "add_to": "功能缺失",
        },
        "max_pool3d_with_indices": {
            "remove_from": ["返回参数类型不一致", "API 别名"],
            "add_to": "功能缺失",
        },
        "range": {
            "remove_from": ["paddle 参数更多", "API 别名"],
            "add_to": "功能缺失",
        },
    }

    # Apply fixes
    apis_added_to_missing = []

    for api_name, fix in fixes.items():
        # Remove from specified sections
        for section_name in fix["remove_from"]:
            if section_name in sections:
                original_count = len(sections[section_name]["rows"])
                sections[section_name]["rows"] = [
                    row
                    for row in sections[section_name]["rows"]
                    if extract_api_name(row) != api_name
                ]
                removed = original_count - len(sections[section_name]["rows"])
                if removed:
                    print(
                        f"Removed {api_name} from '{section_name}' ({removed} row(s))"
                    )

        # Add to missing section if needed
        if fix["add_to"] == "功能缺失":
            apis_added_to_missing.append(api_name)

    # Add APIs to missing section
    if apis_added_to_missing and "功能缺失" in sections:
        for api_name in apis_added_to_missing:
            # Create a new row for missing section
            new_row = f"| - | `at::{api_name}` | - | 功能缺失 | - |"
            sections["功能缺失"]["rows"].append(new_row)
            print(f"Added {api_name} to '功能缺失'")

    # Renumber all sections
    for section_name, section_data in sections.items():
        section_data["rows"] = renumber_rows(section_data["rows"])

    # Rebuild content
    lines = content.split("\n")
    new_lines = []
    current_section = None
    in_table = False
    row_idx = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect section headers
        section_match = re.match(r"### (\d+)\.\s*(.+)", line)
        if section_match:
            current_section = section_match.group(2).strip()
            row_idx = 0
            new_lines.append(line)
            i += 1
            continue

        # Detect table header
        if (
            current_section
            and current_section in sections
            and line.startswith("| 序号 ")
            and "PyTorch C++ API" in line
        ):
            new_lines.append(line)
            i += 1
            # Add separator
            if i < len(lines) and lines[i].startswith("|---"):
                new_lines.append(lines[i])
                i += 1
            # Add all rows from this section
            for row in sections[current_section]["rows"]:
                new_lines.append(row)
            # Skip existing rows until end of table
            while (
                i < len(lines)
                and lines[i].startswith("|")
                and "at::" in lines[i]
            ):
                i += 1
            continue

        new_lines.append(line)
        i += 1

    # Update statistics
    updated_content = "\n".join(new_lines)

    # Find and update the statistics section
    # Count items in each category
    stats = {}
    for section_name, section_data in sections.items():
        stats[section_name] = len(section_data["rows"])

    # Replace statistics
    stat_patterns = [
        (
            r"- \*\*API 完全一致\*\*: \d+ 个",
            f"- **API 完全一致**: {stats.get('API 完全一致', 0)} 个",
        ),
        (
            r"- \*\*仅 API 调用方式不一致\*\*: \d+ 个",
            f"- **仅 API 调用方式不一致**: {stats.get('仅 API 调用方式不一致', 0)} 个",
        ),
        (
            r"- \*\*仅参数名不一致\*\*: \d+ 个",
            f"- **仅参数名不一致**: {stats.get('仅参数名不一致', 0)} 个",
        ),
        (
            r"- \*\*paddle 参数更多\*\*: \d+ 个",
            f"- **paddle 参数更多**: {stats.get('paddle 参数更多', 0)} 个",
        ),
        (
            r"- \*\*参数默认值不一致\*\*: \d+ 个",
            f"- **参数默认值不一致**: {stats.get('参数默认值不一致', 0)} 个",
        ),
        (
            r"- \*\*torch 参数更多\*\*: \d+ 个",
            f"- **torch 参数更多**: {stats.get('torch 参数更多', 0)} 个",
        ),
        (
            r"- \*\*输入参数用法不一致\*\*: \d+ 个",
            f"- **输入参数用法不一致**: {stats.get('输入参数用法不一致', 0)} 个",
        ),
        (
            r"- \*\*输入参数类型不一致\*\*: \d+ 个",
            f"- **输入参数类型不一致**: {stats.get('输入参数类型不一致', 0)} 个",
        ),
        (
            r"- \*\*返回参数类型不一致\*\*: \d+ 个",
            f"- **返回参数类型不一致**: {stats.get('返回参数类型不一致', 0)} 个",
        ),
        (
            r"- \*\*组合替代实现\*\*: \d+ 个",
            f"- **组合替代实现**: {stats.get('组合替代实现', 0)} 个",
        ),
        (
            r"- \*\*API 别名\*\*: \d+ 个",
            f"- **API 别名**: {stats.get('API 别名', 0)} 个",
        ),
        (
            r"- \*\*语义差异\*\*: \d+ 个",
            f"- **语义差异**: {stats.get('语义差异', 0)} 个",
        ),
        (
            r"- \*\*功能缺失\*\*: \d+ 个",
            f"- **功能缺失**: {stats.get('功能缺失', 0)} 个",
        ),
        (r"- \*\*API 别名映射数\*\*: \d+ 个", "- **API 别名映射数**: 14 个"),
    ]

    for pattern, replacement in stat_patterns:
        updated_content = re.sub(pattern, replacement, updated_content)

    # Write back
    with open("cpp_api_mapping_cn.md", "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("\nFix complete!")
    print("\nUpdated statistics:")
    for name, count in sorted(stats.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
