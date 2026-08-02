"""Project style guarantees enforced by the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Declared by Unicode codepoint rather than as literals, so this file does not
# itself contain the characters it forbids. (It did, and failed its own check.)
#
#   U+2014 em dash        U+2013 en dash       U+2012 figure dash
#   U+2015 horizontal bar U+2010 hyphen        U+2011 non-breaking hyphen
#   U+2212 minus sign
#
# The project uses the plain ASCII hyphen everywhere instead.
FORBIDDEN_DASHES = "".join(
    chr(code) for code in (0x2014, 0x2013, 0x2012, 0x2015, 0x2010, 0x2011, 0x2212)
)

SEARCH_DIRS = ("src", "tests", "scripts", "dbt", "orchestration", "deploy", "docs", ".github")
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".sql", ".txt", ".toml", ".cfg", ".ini", ".sh", ".ps1"}


def _text_files() -> list[Path]:
    files: list[Path] = []
    for name in SEARCH_DIRS:
        directory = REPO_ROOT / name
        if directory.is_dir():
            files += [
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
                and "__pycache__" not in p.parts and "egg-info" not in str(p)
            ]
    files += [p for p in REPO_ROOT.glob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES]
    return sorted(set(files))


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: p.name)
def test_no_unicode_dashes(path: Path):
    """Only the ASCII hyphen is permitted anywhere in the project."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip(f"{path.name} is not UTF-8 text")

    offenders = [
        f"  line {n}: {line.strip()[:100]}"
        for n, line in enumerate(content.splitlines(), 1)
        if any(d in line for d in FORBIDDEN_DASHES)
    ]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} contains non-ASCII dash characters. "
        f"Use a plain hyphen '-' instead:\n" + "\n".join(offenders)
    )
