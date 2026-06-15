"""HSA (Hermes Skill Auditor) — 25 structural + security rules as upskill validator.

Usage in eval YAML:
    verifiers:
      - type: validator
        name: hsa-skill-check
        config:
          strict: false
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from upskill.validators import register_validator

if TYPE_CHECKING:
    from upskill.models import ValidationResult


@register_validator("hsa-skill-check")
def hsa_skill_check(workspace: Path, output_file: str = "SKILL.md", **config) -> "ValidationResult":
    """Run HSA structural + security rules against a SKILL.md in workspace.

    Args:
        workspace: The agent's workspace directory.
        output_file: Path to SKILL.md relative to workspace (default: SKILL.md).
        **config: Optional — strict: bool, skip_patterns: list[str].

    Returns:
        ValidationResult with pass/fail and per-rule details.
    """
    from upskill.models import ValidationResult

    skill_path = workspace / output_file
    if not skill_path.exists():
        return ValidationResult(
            passed=False,
            assertions_passed=0,
            assertions_total=25,
            error_message=f"SKILL.md not found at {skill_path}",
            details=["File not found"],
        )

    content = skill_path.read_text(encoding="utf-8", errors="replace")
    strict = config.get("strict", False)

    results = []
    results.extend(_run_structural_rules(content, strict))
    results.extend(_run_security_rules(content))

    passed_count = sum(1 for r in results if r["pass"])
    total = len(results)
    all_pass = passed_count == total

    return ValidationResult(
        passed=all_pass if strict else passed_count >= total - 2,
        assertions_passed=passed_count,
        assertions_total=total,
        details=[
            f"{'✅' if r['pass'] else '❌'} R{r['id']} {r['name']}: {r['detail']}"
            for r in results
        ],
    )


# ═══════════════════════════════════════════
#  Structural Rules (R1-R15, 60 points)
# ═══════════════════════════════════════════


def _parse_frontmatter(content: str) -> dict[str, str]:
    """Extract YAML frontmatter as a dict of key: value."""
    if not content.startswith("---"):
        return {}
    m = re.search(r"\n---\s*\n", content[3:])
    if not m:
        # Unclosed frontmatter — parse everything until first empty line
        end = content.find("\n\n", 3)
        if end == -1:
            end = len(content)
    else:
        end = 3 + m.start()
    frontmatter_text = content[3:end].strip()
    meta = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta


def _run_structural_rules(content: str, strict: bool) -> list[dict]:
    """Run all 15 structural rules."""
    meta = _parse_frontmatter(content)
    body = content.split("---\n", 2)[-1].strip() if content.count("---") >= 2 else ""
    results = []

    # R1: frontmatter-exists
    ok = content.startswith("---")
    results.append({"id": 1, "name": "frontmatter-exists", "pass": ok,
                    "detail": "OK" if ok else "文件不以 --- 开头"})

    # R2: frontmatter-closed
    ok2 = content.count("---") >= 2 and content[3:].find("---") > 0
    results.append({"id": 2, "name": "frontmatter-closed", "pass": ok2,
                    "detail": "OK" if ok2 else "frontmatter 缺少闭合 ---"})

    # R3: name-required
    has_name = "name" in meta
    results.append({"id": 3, "name": "name-required", "pass": has_name,
                    "detail": "OK" if has_name else "缺少 name 字段"})

    # R4: name-format (lowercase-hyphens, ≤64)
    name_val = meta.get("name", "")
    ok4 = bool(re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", name_val))
    results.append({"id": 4, "name": "name-format", "pass": ok4,
                    "detail": "OK" if ok4 else f"name 格式不正确: '{name_val}'"})

    # R5: description-required
    has_desc = "description" in meta
    results.append({"id": 5, "name": "description-required", "pass": has_desc,
                    "detail": "OK" if has_desc else "缺少 description 字段"})

    # R6: description-length (≤1024 chars)
    desc_val = meta.get("description", "")
    ok6 = len(desc_val) <= 1024
    results.append({"id": 6, "name": "description-length", "pass": ok6,
                    "detail": "OK" if ok6 else f"description 过长 ({len(desc_val)} > 1024)"})

    # R7: description-trigger (contains when-to-use keywords)
    ok7 = not has_desc or any(w in desc_val.lower() for w in ["when", "use", "适用", "用于", "触发", "trigger"])
    results.append({"id": 7, "name": "description-trigger", "pass": ok7,
                    "detail": "OK" if ok7 else "description 缺少触发条件关键词"})

    # R8: body-exists (≥50 chars after frontmatter)
    ok8 = len(body) >= 50
    results.append({"id": 8, "name": "body-exists", "pass": ok8,
                    "detail": "OK" if ok8 else f"正文不足 50 字符 (实际 {len(body)})"})

    # R9: size-limit (≤100K chars)
    ok9 = len(content) <= 100_000
    results.append({"id": 9, "name": "size-limit", "pass": ok9,
                    "detail": "OK" if ok9 else f"文件过大 ({len(content)} > 100000)"})

    # R10: version-present
    has_ver = "version" in meta
    results.append({"id": 10, "name": "version-present", "pass": has_ver,
                    "detail": "OK" if has_ver else "缺少 version 字段"})

    # R11: license-present
    has_lic = "license" in meta
    results.append({"id": 11, "name": "license-present", "pass": has_lic,
                    "detail": "OK" if has_lic else "缺少 license 字段"})

    # R12: tags-present
    has_tags = "tags" in meta
    results.append({"id": 12, "name": "tags-present", "pass": has_tags,
                    "detail": "OK" if has_tags else "缺少 tags 字段"})

    # R13: body-sections (has at least section headers)
    has_sections = bool(re.search(r"^#{1,3}\s+", body, re.MULTILINE))
    results.append({"id": 13, "name": "body-sections", "pass": has_sections,
                    "detail": "OK" if has_sections else "正文缺少章节标题"})

    # R14: no-leading-blank
    ok14 = not content.startswith("\n")
    results.append({"id": 14, "name": "no-leading-blank", "pass": ok14,
                    "detail": "OK" if ok14 else "文件以空行开头"})

    # R15: no-trailing-whitespace
    ok15 = not re.search(r"[ \t]+$", content, re.MULTILINE)
    results.append({"id": 15, "name": "no-trailing-whitespace", "pass": ok15,
                    "detail": "OK" if ok15 else "存在行尾空白"})

    return results


# ═══════════════════════════════════════════
#  Security Rules (R16-R20, 10 points)
# ═══════════════════════════════════════════


def _run_security_rules(content: str) -> list[dict]:
    """Run all 5 security rules."""
    results = []

    # R16: no-secrets (API keys, tokens, passwords)
    secret_patterns = [
        r'(?:api[_-]?key|apikey|secret|token|password)\s*[:=]\s*[\'"][^\'"]{10,}[\'"]',
        r'sk-[a-zA-Z0-9]{20,}',
        r'AKIA[0-9A-Z]{16}',
        r'ghp_[a-zA-Z0-9]{36,}',
        r'eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}',
    ]
    found_secrets = []
    for pat in secret_patterns:
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            found_secrets.extend(matches[:2])
    ok_s = len(found_secrets) == 0
    results.append({"id": 16, "name": "no-secrets", "pass": ok_s,
                    "detail": "OK" if ok_s else f"发现疑似密钥 ({len(found_secrets)}处)"})

    # R17: safe-commands (no rm -rf /, fork bombs, curl|bash)
    dangerous = [
        (r'rm\s+-rf\s+/', "rm -rf /"),
        (r':\(\)\s*\{\s*:\|:&\s*\};\s*:', "fork bomb"),
        (r'>\s*/dev/sda', "overwrite disk"),
        (r'curl\s+.*\|\s*(?:ba)?sh', "curl pipe to shell"),
        (r'wget\s+.*\s*-O\s*-\s*\|\s*(?:ba)?sh', "wget pipe to shell"),
    ]
    found_danger = []
    for pat, desc in dangerous:
        if re.search(pat, content):
            found_danger.append(desc)
    ok_d = len(found_danger) == 0
    results.append({"id": 17, "name": "safe-commands", "pass": ok_d,
                    "detail": "OK" if ok_d else f"危险命令: {', '.join(found_danger)}"})

    # R18: no-phishing-urls
    urls = re.findall(r'https?://[^\s\)\]>"\']+', content)
    known_sus = ['githib.com', 'githab.com', 'discord.gift', 'tinyurl.com']
    sus_urls = [u for u in urls if any(s in u.lower() for s in known_sus)]
    ok_u = len(sus_urls) == 0
    results.append({"id": 18, "name": "no-phishing-urls", "pass": ok_u,
                    "detail": "OK" if ok_u else f"可疑URL: {sus_urls[0]}"})

    # R19: script-safety (no eval/exec in code blocks)
    evals = re.findall(r'\b(eval|exec)\s*\(', content)
    ok_e = len(evals) == 0
    results.append({"id": 19, "name": "script-safety", "pass": ok_e,
                    "detail": "OK" if ok_e else f"发现 eval/exec 调用 ({len(evals)}处)"})

    # R20: no-hardcoded-paths (/home/user, C:\Users\)
    path_patterns = [
        r'/home/\w+',
        r'C:\\Users\\\w+',
        r'/Users/\w+',
    ]
    found_paths = []
    for pp in path_patterns:
        matches = re.findall(pp, content)
        if matches:
            found_paths.extend(matches[:2])
    ok_p = len(found_paths) == 0
    results.append({"id": 20, "name": "no-hardcoded-paths", "pass": ok_p,
                    "detail": "OK" if ok_p else f"硬编码用户路径 ({len(found_paths)}处)"})

    return results
