"""Validate local Markdown links and JSON examples without external services."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [ROOT / "README.md", *sorted((ROOT / "doc").glob("*.md"))]
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def anchors(path: Path) -> set[str]:
    """Return the section anchors GitHub generates for ordinary ATX headings."""
    found: set[str] = set()
    duplicates: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match is None:
            continue
        title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", match.group(1))
        title = re.sub(r"<[^>]+>", "", title).replace("`", "")
        slug = re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())
        duplicate_number = duplicates[slug]
        duplicates[slug] += 1
        found.add(f"{slug}-{duplicate_number}" if duplicate_number else slug)
    return found


def check_markdown(path: Path) -> list[str]:
    problems: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for target in LINK.findall(line):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            local_path = unquote(parsed.path)
            destination = (path.parent / local_path).resolve() if local_path else path
            display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            location = f"{display_path}:{line_number}"
            if not destination.exists():
                problems.append(f"{location}: missing target {target}")
            elif parsed.fragment and destination.suffix.lower() == ".md":
                anchor = unquote(parsed.fragment)
                if anchor not in anchors(destination):
                    problems.append(f"{location}: missing section {target}")
    return problems


def main() -> int:
    problems = [issue for path in MARKDOWN_FILES for issue in check_markdown(path)]
    for path in sorted((ROOT / "examples").glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            problems.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Checked {len(MARKDOWN_FILES)} Markdown files and example JSON files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
