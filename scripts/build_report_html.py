"""Render a markdown report to a single self-contained HTML file (images embedded).

Converts the markdown to HTML and inlines every ``figures/*.png`` reference as a base64
data URI, so the output is one portable file a layperson can open in any browser with no
missing-image issues. Default target: the layperson draft-pool report.

Usage: uv run python scripts/build_report_html.py [path/to/report.md]
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "docs" / "report_draft_pool_layperson_2026-06-08.md"

CSS = """
:root { color-scheme: light; }
body { max-width: 820px; margin: 2.2rem auto; padding: 0 1.1rem;
  font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  font-size: 17px; line-height: 1.6; color: #1a1a1a; background: #fff; }
h1 { font-size: 2rem; line-height: 1.25; border-bottom: 3px solid #2b6cb0; padding-bottom: .3rem; }
h2 { font-size: 1.45rem; margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
h3 { font-size: 1.15rem; margin-top: 1.5rem; color: #2b3a4a; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 15.5px; }
th, td { border: 1px solid #cbd5e0; padding: .5rem .6rem; text-align: left; vertical-align: top; }
th { background: #edf2f7; }
tr:nth-child(even) td { background: #f7fafc; }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto;
  border: 1px solid #e2e8f0; border-radius: 6px; }
code { background: #edf2f7; padding: .08em .35em; border-radius: 4px; font-size: .92em; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
blockquote { color: #555; border-left: 4px solid #cbd5e0; margin: 1rem 0; padding: .2rem 1rem; }
a { color: #2b6cb0; }
""".strip()

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body>{body}</body></html>
"""


def _inline_images(html: str, base_dir: Path) -> str:
    """Replace src="figures/x.png" with a base64 data URI of the file's bytes."""

    def repl(match: re.Match) -> str:
        rel = match.group(1)
        path = (base_dir / rel).resolve()
        if not path.is_file():
            return match.group(0)  # leave untouched if missing
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="((?:\.\./)*figures/[^"]+\.png)"', repl, html)


def build(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    body = _inline_images(body, md_path.parent)
    headings = (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# "))
    title = next(headings, "Report")
    html = TEMPLATE.format(title=title, css=CSS, body=body)
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    out = build(md_path)
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out.relative_to(ROOT)} ({size_kb:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
