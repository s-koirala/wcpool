"""Render the report to PDF by printing its self-contained HTML via headless Chromium.

Rebuilds the HTML (figures inlined) from the markdown, then drives an installed Chromium
browser (Edge or Chrome) in headless mode to print it to PDF. No extra Python deps and no
LaTeX needed. Falls back across known Windows browser locations and PATH.

Usage: uv run python scripts/build_report_pdf.py [path/to/report.md]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report_html import DEFAULT_MD, build  # noqa: E402

# Candidate Chromium browsers: PATH names first, then known Windows install locations.
_BROWSERS = ["chrome", "chromium", "msedge"]
_WIN_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    for p in _WIN_PATHS:
        if Path(p).is_file():
            return p
    raise SystemExit("no Chromium browser (Chrome/Edge) found to render the PDF")


def html_to_pdf(html_path: Path, browser: str) -> Path:
    pdf_path = html_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(prefix="wcpool_pdf_") as profile:
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise SystemExit("browser ran but produced no PDF")
    return pdf_path


def main() -> None:
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    html_path = build(md_path)  # rebuild HTML so the PDF reflects the latest markdown
    browser = find_browser()
    pdf_path = html_to_pdf(html_path, browser)
    root = Path(__file__).resolve().parents[1]
    size_kb = pdf_path.stat().st_size / 1024
    print(f"wrote {pdf_path.relative_to(root)} ({size_kb:.0f} KB) via {Path(browser).name}")


if __name__ == "__main__":
    main()
