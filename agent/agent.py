#!/usr/bin/env python3
"""
Tech Debt Agent - 一个轻量级代码库维护 / 技术债扫描 Agent

功能：
1. 扫描代码库文件
2. 用规则引擎发现技术债、坏味道、重复代码、潜在风险
3. 可选调用 OpenAI-compatible LLM 做深度分析
4. 生成 Markdown 报告
5. 可选生成补丁建议，但默认不直接修改代码

使用方式：
    python tech_debt_agent.py --repo /path/to/repo --out report.md
    python tech_debt_agent.py --repo . --llm --out report.md

环境变量：
    OPENAI_API_KEY       可选，用于 LLM 分析
    OPENAI_BASE_URL      可选，默认 https://api.openai.com/v1
    OPENAI_MODEL         可选，默认 gpt-4o-mini

依赖：
    Python 3.10+
    pip install requests
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError:
    requests = None


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".php", ".rb",
    ".cs", ".cpp", ".c", ".h", ".hpp", ".kt", ".swift", ".vue", ".svelte",
}

DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules", "dist", "build",
    "coverage", ".next", ".nuxt", ".turbo", ".cache", "target", "vendor", "__pycache__",
    ".venv", "venv", "env", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}

RISK_WEIGHTS = {
    "critical": 10,
    "high": 7,
    "medium": 4,
    "low": 2,
    "info": 1,
}


@dataclasses.dataclass
class FileInfo:
    path: Path
    relative_path: str
    extension: str
    text: str
    lines: list[str]
    size_bytes: int


@dataclasses.dataclass
class Finding:
    rule_id: str
    title: str
    severity: str
    file_path: str
    line: int | None
    message: str
    suggestion: str
    evidence: str = ""
    auto_fix: str | None = None

    @property
    def risk_score(self) -> int:
        return RISK_WEIGHTS.get(self.severity, 1)


@dataclasses.dataclass
class ScanSummary:
    total_files: int
    total_lines: int
    findings_count: int
    severity_counts: dict[str, int]
    top_risky_files: list[tuple[str, int]]
    duplicate_groups: int
    elapsed_seconds: float


class RepoScanner:
    def __init__(
        self,
        repo: Path,
        max_file_bytes: int = 300_000,
        ignore_dirs: set[str] | None = None,
        include_extensions: set[str] | None = None,
    ) -> None:
        self.repo = repo.resolve()
        self.max_file_bytes = max_file_bytes
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
        self.include_extensions = include_extensions or SUPPORTED_EXTENSIONS

    def scan(self) -> list[FileInfo]:
        files: list[FileInfo] = []
        for path in self.repo.rglob("*"):
            if not path.is_file():
                continue
            if self._is_ignored(path):
                continue
            if path.suffix.lower() not in self.include_extensions:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self.max_file_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    text = path.read_text(encoding="latin-1")
                except Exception:
                    continue
            except Exception:
                continue

            rel = str(path.relative_to(self.repo)).replace(os.sep, "/")
            files.append(
                FileInfo(
                    path=path,
                    relative_path=rel,
                    extension=path.suffix.lower(),
                    text=text,
                    lines=text.splitlines(),
                    size_bytes=size,
                )
            )
        return files

    def _is_ignored(self, path: Path) -> bool:
        parts = set(path.relative_to(self.repo).parts)
        return bool(parts & self.ignore_dirs)


class RuleEngine:
    def __init__(self) -> None:
        self.rules = [
            self.rule_todo_fixme,
            self.rule_console_or_print,
            self.rule_hardcoded_secret,
            self.rule_long_file,
            self.rule_long_function_python,
            self.rule_broad_exception_python,
            self.rule_deep_nesting,
            self.rule_magic_numbers,
            self.rule_large_comment_debt,
            self.rule_missing_tests_hint,
        ]

    def analyze(self, files: list[FileInfo]) -> list[Finding]:
        findings: list[Finding] = []
        for file in files:
            for rule in self.rules:
                try:
                    findings.extend(rule(file))
                except Exception as exc:
                    findings.append(
                        Finding(
                            rule_id="rule_error",
                            title="规则执行失败",
                            severity="info",
                            file_path=file.relative_path,
                            line=None,
                            message=f"某条扫描规则执行失败：{exc}",
                            suggestion="检查该文件是否包含非常规语法或编码。",
                        )
                    )
        findings.extend(self.rule_duplicate_blocks(files))
        return findings

    def rule_todo_fixme(self, file: FileInfo) -> list[Finding]:
        pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX|TEMP|WORKAROUND)\b[:：]?\s*(.*)", re.I)
        results: list[Finding] = []
        for i, line in enumerate(file.lines, start=1):
            match = pattern.search(line)
            if match:
                tag = match.group(1).upper()
                message = match.group(2).strip()[:160]
                results.append(
                    Finding(
                        rule_id="debt_marker",
                        title="遗留技术债标记",
                        severity="medium" if tag in {"FIXME", "HACK", "WORKAROUND"} else "low",
                        file_path=file.relative_path,
                        line=i,
                        message=f"发现 {tag} 标记：{message or '未填写说明'}",
                        suggestion="将该技术债转为 issue，补充 owner、截止时间和验收标准；若已经过期，应优先清理。",
                        evidence=line.strip(),
                    )
                )
        return results

    def rule_console_or_print(self, file: FileInfo) -> list[Finding]:
        if file.extension not in {".py", ".js", ".jsx", ".ts", ".tsx"}:
            return []
        patterns = {
            ".py": re.compile(r"(^|\s)print\s*\("),
            ".js": re.compile(r"console\.(log|debug|warn|error)\s*\("),
            ".jsx": re.compile(r"console\.(log|debug|warn|error)\s*\("),
            ".ts": re.compile(r"console\.(log|debug|warn|error)\s*\("),
            ".tsx": re.compile(r"console\.(log|debug|warn|error)\s*\("),
        }
        regex = patterns[file.extension]
        results = []
        for i, line in enumerate(file.lines, start=1):
            if regex.search(line):
                results.append(
                    Finding(
                        rule_id="debug_output",
                        title="疑似调试输出残留",
                        severity="low",
                        file_path=file.relative_path,
                        line=i,
                        message="发现 print 或 console 输出，可能是调试残留。",
                        suggestion="生产代码建议改为结构化 logger，并根据环境控制日志级别。",
                        evidence=line.strip(),
                    )
                )
        return results

    def rule_hardcoded_secret(self, file: FileInfo) -> list[Finding]:
        secret_patterns = [
            re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
            re.compile(r"AKIA[0-9A-Z]{16}"),
            re.compile(r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"),
            re.compile(r"(?i)bearer\s+[a-z0-9_\-.=]{20,}"),
        ]
        results = []
        for i, line in enumerate(file.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            for pattern in secret_patterns:
                if pattern.search(line):
                    masked = re.sub(r"(['\"])[^'\"]{8,}(['\"])", r"\1***MASKED***\2", stripped)
                    results.append(
                        Finding(
                            rule_id="hardcoded_secret",
                            title="疑似硬编码密钥或密码",
                            severity="critical",
                            file_path=file.relative_path,
                            line=i,
                            message="发现疑似密钥、token、密码或私钥。",
                            suggestion="立即移除硬编码凭证，改用环境变量或密钥管理服务；如果已提交到仓库，需要轮换密钥。",
                            evidence=masked,
                        )
                    )
                    break
        return results

    def rule_long_file(self, file: FileInfo) -> list[Finding]:
        line_count = len(file.lines)
        if line_count < 500:
            return []
        severity = "high" if line_count >= 1200 else "medium"
        return [
            Finding(
                rule_id="long_file",
                title="文件过长",
                severity=severity,
                file_path=file.relative_path,
                line=None,
                message=f"文件共有 {line_count} 行，维护成本较高。",
                suggestion="按职责拆分模块；优先抽离纯函数、常量、类型定义、数据访问层或 UI 子组件。",
            )
        ]

    def rule_long_function_python(self, file: FileInfo) -> list[Finding]:
        if file.extension != ".py":
            return []
        try:
            tree = ast.parse(file.text)
        except SyntaxError:
            return []
        results: list[Finding] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start and end:
                    length = end - start + 1
                    if length >= 80:
                        results.append(
                            Finding(
                                rule_id="long_function",
                                title="Python 函数过长",
                                severity="high" if length >= 150 else "medium",
                                file_path=file.relative_path,
                                line=start,
                                message=f"函数 `{node.name}` 共有 {length} 行，可能承担过多职责。",
                                suggestion="将函数拆成较小的步骤函数；为复杂条件添加单元测试后再重构。",
                                evidence=f"def {node.name}(...): {length} lines",
                            )
                        )
        return results

    def rule_broad_exception_python(self, file: FileInfo) -> list[Finding]:
        if file.extension != ".py":
            return []
        results = []
        for i, line in enumerate(file.lines, start=1):
            if re.search(r"except\s+(Exception|BaseException)?\s*:", line):
                results.append(
                    Finding(
                        rule_id="broad_exception",
                        title="过宽异常捕获",
                        severity="medium",
                        file_path=file.relative_path,
                        line=i,
                        message="发现宽泛的 except，可能吞掉真实错误。",
                        suggestion="捕获具体异常类型；至少记录异常上下文并避免静默失败。",
                        evidence=line.strip(),
                    )
                )
        return results

    def rule_deep_nesting(self, file: FileInfo) -> list[Finding]:
        results = []
        for i, line in enumerate(file.lines, start=1):
            if not line.strip():
                continue
            indent_spaces = len(line) - len(line.lstrip(" "))
            if "\t" in line[: max(1, len(line) - len(line.lstrip()))]:
                indent_level = line.count("\t")
            else:
                indent_level = indent_spaces // 4
            if indent_level >= 6:
                results.append(
                    Finding(
                        rule_id="deep_nesting",
                        title="嵌套层级过深",
                        severity="medium",
                        file_path=file.relative_path,
                        line=i,
                        message=f"该行缩进层级约为 {indent_level}，可读性和可测试性较差。",
                        suggestion="使用 guard clause、提前 return、策略模式或拆分函数降低嵌套。",
                        evidence=line.strip()[:180],
                    )
                )
                break
        return results

    def rule_magic_numbers(self, file: FileInfo) -> list[Finding]:
        if file.extension not in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cs"}:
            return []
        ignore_numbers = {"0", "1", "2", "100", "200", "201", "204", "400", "401", "403", "404", "500"}
        number_pattern = re.compile(r"(?<![\w.])-?\d{2,}(?:\.\d+)?(?![\w.])")
        found: list[tuple[int, str]] = []
        for i, line in enumerate(file.lines, start=1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "/*")):
                continue
            matches = number_pattern.findall(line)
            filtered = [m for m in matches if m not in ignore_numbers]
            if filtered:
                found.append((i, stripped[:180]))
        if len(found) < 8:
            return []
        line, evidence = found[0]
        return [
            Finding(
                rule_id="magic_numbers",
                title="魔法数字过多",
                severity="low",
                file_path=file.relative_path,
                line=line,
                message=f"文件中发现较多未命名数字常量，数量约 {len(found)} 处。",
                suggestion="将业务含义明确的数字抽为命名常量或配置项。",
                evidence=evidence,
            )
        ]

    def rule_large_comment_debt(self, file: FileInfo) -> list[Finding]:
        comment_lines = 0
        code_lines = 0
        for line in file.lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", "//", "/*", "*")):
                comment_lines += 1
            else:
                code_lines += 1
        total = comment_lines + code_lines
        if total < 100:
            return []
        ratio = comment_lines / max(total, 1)
        if ratio <= 0.45:
            return []
        return [
            Finding(
                rule_id="comment_heavy",
                title="注释占比异常偏高",
                severity="info",
                file_path=file.relative_path,
                line=None,
                message=f"注释占比约 {ratio:.0%}，可能存在废弃代码、过度解释或历史遗留逻辑。",
                suggestion="检查是否有大段注释掉的旧代码；将关键设计说明迁移到文档或 ADR。",
            )
        ]

    def rule_missing_tests_hint(self, file: FileInfo) -> list[Finding]:
        path = file.relative_path.lower()
        if any(part in path for part in ["test", "spec", "__tests__"]):
            return []
        risky_keywords = ["payment", "auth", "permission", "billing", "refund", "crypto", "security"]
        if not any(k in path for k in risky_keywords):
            return []
        return [
            Finding(
                rule_id="risky_area_test_hint",
                title="高风险领域建议补充测试",
                severity="medium",
                file_path=file.relative_path,
                line=None,
                message="文件路径显示它可能涉及支付、权限、安全或账务等高风险领域。",
                suggestion="确认是否存在对应单元测试和集成测试，重点覆盖失败、重试、权限边界和幂等场景。",
            )
        ]

    def rule_duplicate_blocks(self, files: list[FileInfo]) -> list[Finding]:
        block_map: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        min_block_lines = 8

        for file in files:
            normalized_lines = []
            original_indexes = []
            for i, line in enumerate(file.lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(("#", "//", "/*", "*")):
                    continue
                normalized = re.sub(r"\s+", " ", stripped)
                normalized = re.sub(r"['\"][^'\"]{1,80}['\"]", "<str>", normalized)
                normalized_lines.append(normalized)
                original_indexes.append(i)

            for idx in range(0, len(normalized_lines) - min_block_lines + 1):
                block = "\n".join(normalized_lines[idx : idx + min_block_lines])
                digest = hashlib.sha1(block.encode("utf-8")).hexdigest()
                start_line = original_indexes[idx]
                block_map[digest].append((file.relative_path, start_line, block))

        results: list[Finding] = []
        for digest, occurrences in block_map.items():
            unique_locations = {(p, line) for p, line, _ in occurrences}
            unique_files = {p for p, _, _ in occurrences}
            if len(unique_locations) < 2 or len(unique_files) < 2:
                continue
            sample = occurrences[0][2].splitlines()[0][:160]
            locations = sorted(unique_locations)[:5]
            location_text = ", ".join(f"{p}:{line}" for p, line in locations)
            first_file, first_line, _ = occurrences[0]
            results.append(
                Finding(
                    rule_id="duplicate_block",
                    title="跨文件重复代码块",
                    severity="medium",
                    file_path=first_file,
                    line=first_line,
                    message=f"发现至少 {len(unique_locations)} 处相似代码块：{location_text}",
                    suggestion="考虑抽取公共函数、工具模块、基类、hook 或共享组件；重构前先补测试保证行为一致。",
                    evidence=sample,
                )
            )
        return results[:50]


class GitInspector:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def is_git_repo(self) -> bool:
        result = self._run(["git", "rev-parse", "--is-inside-work-tree"])
        return result.strip() == "true"

    def recent_hotspots(self, max_files: int = 20) -> list[tuple[str, int]]:
        if not self.is_git_repo():
            return []
        output = self._run(["git", "log", "--name-only", "--pretty=format:", "--since=180 days"])
        counter: Counter[str] = Counter()
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith("commit "):
                counter[line] += 1
        return counter.most_common(max_files)

    def current_branch(self) -> str:
        if not self.is_git_repo():
            return "not-a-git-repo"
        branch = self._run(["git", "branch", "--show-current"])
        return branch.strip() or "unknown"

    def _run(self, command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.repo),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=20,
            )
            return completed.stdout
        except Exception:
            return ""


class LLMReviewer:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if requests is None:
            raise RuntimeError("缺少 requests 依赖，请运行：pip install requests")
        if not self.api_key:
            raise RuntimeError("未设置 OPENAI_API_KEY，无法启用 LLM 分析")

    def review_file(self, file: FileInfo, local_findings: list[Finding]) -> list[Finding]:
        if len(file.text) > 20_000:
            content = file.text[:20_000] + "\n... TRUNCATED ..."
        else:
            content = file.text

        local_summary = [
            {
                "rule_id": f.rule_id,
                "severity": f.severity,
                "line": f.line,
                "message": f.message,
            }
            for f in local_findings[:20]
        ]

        prompt = f"""
你是一个资深代码审查 Agent。请分析下面文件的技术债、潜在 bug、可维护性问题。

要求：
1. 只输出 JSON 数组，不要 markdown。
2. 每个对象字段必须包含：title, severity, line, message, suggestion, evidence。
3. severity 只能是 critical/high/medium/low/info。
4. 最多输出 5 个最重要的问题。
5. 不要编造不存在的行号；不确定行号时填 null。
6. 避免重复本地规则已发现的问题，除非你能给出更深的判断。

文件路径：{file.relative_path}
本地规则发现：{json.dumps(local_summary, ensure_ascii=False)}

代码内容：
```{file.extension.lstrip('.')}
{content}
```
""".strip()

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是严谨的代码库维护与技术债分析助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60,
        )
        response.raise_for_status()
        content_text = response.json()["choices"][0]["message"]["content"]
        parsed = self._parse_json_array(content_text)
        findings = []
        for item in parsed:
            findings.append(
                Finding(
                    rule_id="llm_review",
                    title=str(item.get("title", "LLM 深度审查发现"))[:120],
                    severity=self._safe_severity(item.get("severity")),
                    file_path=file.relative_path,
                    line=item.get("line") if isinstance(item.get("line"), int) else None,
                    message=str(item.get("message", ""))[:500],
                    suggestion=str(item.get("suggestion", ""))[:500],
                    evidence=str(item.get("evidence", ""))[:300],
                )
            )
        return findings

    def _parse_json_array(self, text: str) -> list[dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", cleaned, flags=re.S)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []

    def _safe_severity(self, value: Any) -> str:
        value = str(value).lower()
        return value if value in RISK_WEIGHTS else "medium"


class PatchSuggester:
    """生成保守的本地补丁建议。默认只生成 diff，不直接写入文件。"""

    def suggest_patch_for_debug_output(self, file: FileInfo) -> str | None:
        if file.extension != ".py":
            return None
        changed = False
        new_lines = file.lines[:]
        needs_logging_import = False
        for idx, line in enumerate(new_lines):
            if re.search(r"(^|\s)print\s*\(", line):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines[idx] = indent + "logging.info(" + re.sub(r"^\s*print\s*\(", "", line).rstrip()
                changed = True
                needs_logging_import = True
        if needs_logging_import and not any(re.match(r"\s*import\s+logging\b", line) for line in new_lines[:30]):
            insert_at = 0
            while insert_at < len(new_lines) and new_lines[insert_at].startswith(("#!", "# -*-")):
                insert_at += 1
            new_lines.insert(insert_at, "import logging")
        if not changed:
            return None
        return self._unified_diff(file.lines, new_lines, file.relative_path)

    def _unified_diff(self, old: list[str], new: list[str], path: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )


class ReportBuilder:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def build(
        self,
        files: list[FileInfo],
        findings: list[Finding],
        git_hotspots: list[tuple[str, int]],
        elapsed_seconds: float,
        patches: dict[str, str] | None = None,
    ) -> str:
        patches = patches or {}
        severity_counts = Counter(f.severity for f in findings)
        risk_by_file: Counter[str] = Counter()
        for f in findings:
            risk_by_file[f.file_path] += f.risk_score
        duplicate_groups = sum(1 for f in findings if f.rule_id == "duplicate_block")
        summary = ScanSummary(
            total_files=len(files),
            total_lines=sum(len(f.lines) for f in files),
            findings_count=len(findings),
            severity_counts=dict(severity_counts),
            top_risky_files=risk_by_file.most_common(10),
            duplicate_groups=duplicate_groups,
            elapsed_seconds=elapsed_seconds,
        )

        lines: list[str] = []
        lines.append("# 代码库技术债扫描报告")
        lines.append("")
        lines.append(f"- 仓库路径：`{self.repo}`")
        lines.append(f"- 扫描文件数：**{summary.total_files}**")
        lines.append(f"- 扫描代码行数：**{summary.total_lines}**")
        lines.append(f"- 发现问题数：**{summary.findings_count}**")
        lines.append(f"- 重复代码组：**{summary.duplicate_groups}**")
        lines.append(f"- 耗时：**{summary.elapsed_seconds:.2f}s**")
        lines.append("")

        lines.append("## 严重程度分布")
        lines.append("")
        lines.append("| 严重程度 | 数量 |")
        lines.append("|---|---:|")
        for severity in ["critical", "high", "medium", "low", "info"]:
            lines.append(f"| {severity} | {summary.severity_counts.get(severity, 0)} |")
        lines.append("")

        if summary.top_risky_files:
            lines.append("## 风险最高文件 Top 10")
            lines.append("")
            lines.append("| 文件 | 风险分 |")
            lines.append("|---|---:|")
            for path, score in summary.top_risky_files:
                lines.append(f"| `{path}` | {score} |")
            lines.append("")

        if git_hotspots:
            lines.append("## 最近 180 天变更热点")
            lines.append("")
            lines.append("这些文件变更频繁，若同时存在技术债，应优先治理。")
            lines.append("")
            lines.append("| 文件 | 变更次数 |")
            lines.append("|---|---:|")
            for path, count in git_hotspots[:15]:
                lines.append(f"| `{path}` | {count} |")
            lines.append("")

        lines.append("## 问题清单")
        lines.append("")
        sorted_findings = sorted(
            findings,
            key=lambda f: (-f.risk_score, f.file_path, f.line or 0, f.rule_id),
        )
        for idx, f in enumerate(sorted_findings, start=1):
            location = f"{f.file_path}:{f.line}" if f.line else f.file_path
            lines.append(f"### {idx}. [{f.severity.upper()}] {f.title}")
            lines.append("")
            lines.append(f"- 规则：`{f.rule_id}`")
            lines.append(f"- 位置：`{location}`")
            lines.append(f"- 问题：{f.message}")
            lines.append(f"- 建议：{f.suggestion}")
            if f.evidence:
                lines.append("- 证据：")
                lines.append("")
                lines.append("```text")
                lines.append(f.evidence)
                lines.append("```")
            lines.append("")

        if patches:
            lines.append("## 可选补丁建议")
            lines.append("")
            lines.append("以下 diff 仅作为保守建议，默认未自动写入仓库。请人工 review 后再应用。")
            lines.append("")
            for path, patch in patches.items():
                lines.append(f"### {path}")
                lines.append("")
                lines.append("```diff")
                lines.append(patch)
                lines.append("```")
                lines.append("")

        lines.append("## 推荐落地流程")
        lines.append("")
        lines.append("1. 先处理 critical/high 问题，尤其是硬编码密钥、安全和高风险业务逻辑。")
        lines.append("2. 对 Top 风险文件补充测试，再做结构性重构。")
        lines.append("3. 将 TODO/FIXME 转为带 owner 和 deadline 的 issue。")
        lines.append("4. 把本脚本接入 CI，每次 PR 只扫描变更文件，主分支每天全量扫描一次。")
        lines.append("5. 对 LLM 分析启用缓存和文件风险初筛，避免全仓库无差别调用模型。")
        lines.append("")
        return "\n".join(lines)


class TechDebtAgent:
    def __init__(
        self,
        repo: Path,
        use_llm: bool = False,
        llm_limit: int = 8,
        max_file_bytes: int = 300_000,
    ) -> None:
        self.repo = repo.resolve()
        self.use_llm = use_llm
        self.llm_limit = llm_limit
        self.max_file_bytes = max_file_bytes
        self.scanner = RepoScanner(repo=self.repo, max_file_bytes=max_file_bytes)
        self.rule_engine = RuleEngine()
        self.git = GitInspector(self.repo)
        self.report_builder = ReportBuilder(self.repo)
        self.patch_suggester = PatchSuggester()

    def run(self) -> tuple[str, list[Finding]]:
        start = time.time()
        files = self.scanner.scan()
        findings = self.rule_engine.analyze(files)

        if self.use_llm:
            findings.extend(self._run_llm_review(files, findings))

        patches = self._generate_patch_suggestions(files, findings)
        git_hotspots = self.git.recent_hotspots()
        elapsed = time.time() - start
        report = self.report_builder.build(files, findings, git_hotspots, elapsed, patches)
        return report, findings

    def _run_llm_review(self, files: list[FileInfo], findings: list[Finding]) -> list[Finding]:
        local_by_file: dict[str, list[Finding]] = defaultdict(list)
        for finding in findings:
            local_by_file[finding.file_path].append(finding)

        risk_by_file: Counter[str] = Counter()
        for finding in findings:
            risk_by_file[finding.file_path] += finding.risk_score

        candidates = sorted(
            files,
            key=lambda f: (risk_by_file.get(f.relative_path, 0), len(f.lines)),
            reverse=True,
        )[: self.llm_limit]

        reviewer = LLMReviewer()
        llm_findings: list[Finding] = []
        for file in candidates:
            try:
                llm_findings.extend(reviewer.review_file(file, local_by_file.get(file.relative_path, [])))
            except Exception as exc:
                llm_findings.append(
                    Finding(
                        rule_id="llm_error",
                        title="LLM 分析失败",
                        severity="info",
                        file_path=file.relative_path,
                        line=None,
                        message=f"LLM 分析该文件失败：{exc}",
                        suggestion="检查 API key、网络、模型名称或上下文长度限制；本地规则扫描仍然有效。",
                    )
                )
        return llm_findings

    def _generate_patch_suggestions(self, files: list[FileInfo], findings: list[Finding]) -> dict[str, str]:
        files_with_debug_output = {f.file_path for f in findings if f.rule_id == "debug_output"}
        patches: dict[str, str] = {}
        for file in files:
            if file.relative_path not in files_with_debug_output:
                continue
            patch = self.patch_suggester.suggest_patch_for_debug_output(file)
            if patch:
                patches[file.relative_path] = patch
        return patches


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="代码库维护 / 技术债扫描 Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=str, default=".", help="要扫描的代码库路径")
    parser.add_argument("--out", type=str, default="tech_debt_report.md", help="报告输出路径")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 深度审查，需要 OPENAI_API_KEY")
    parser.add_argument("--llm-limit", type=int, default=8, help="最多交给 LLM 深度分析的文件数")
    parser.add_argument("--max-file-bytes", type=int, default=300_000, help="单文件最大扫描字节数")
    parser.add_argument("--fail-on", choices=["none", "critical", "high", "medium"], default="none", help="达到某严重程度时以非零状态码退出，适合 CI")
    return parser


def should_fail(findings: list[Finding], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    threshold = RISK_WEIGHTS[fail_on]
    return any(f.risk_score >= threshold for f in findings)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo = Path(args.repo)
    if not repo.exists() or not repo.is_dir():
        print(f"错误：仓库路径不存在或不是目录：{repo}", file=sys.stderr)
        return 2

    agent = TechDebtAgent(
        repo=repo,
        use_llm=args.llm,
        llm_limit=args.llm_limit,
        max_file_bytes=args.max_file_bytes,
    )
    report, findings = agent.run()

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")

    severity_counts = Counter(f.severity for f in findings)
    print("扫描完成")
    print(f"报告路径：{out_path.resolve()}")
    print(f"问题总数：{len(findings)}")
    print("严重程度：" + ", ".join(f"{k}={severity_counts.get(k, 0)}" for k in ["critical", "high", "medium", "low", "info"]))

    return 1 if should_fail(findings, args.fail_on) else 0


if __name__ == "__main__":
    raise SystemExit(main())
