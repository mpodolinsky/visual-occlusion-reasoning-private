#!/usr/bin/env python3
"""Render a Markdown evaluation report as a print-friendly PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from markdown_it import MarkdownIt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "pi05_libero_comparison" / "occlusion_failures.md"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pi05_libero_comparison" / "occlusion_failures.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="LIBERO-Occ: Episodes That Failed Only With Occlusion")
    return parser.parse_args()


def render_html(markdown_text: str, title: str) -> str:
    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown_text)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: Letter; margin: 0.62in 0.58in 0.68in; }}
  * {{ box-sizing: border-box; }}
  html {{ font-family: Arial, Helvetica, sans-serif; color: #202124; }}
  body {{ margin: 0; font-size: 9.2pt; line-height: 1.38; }}
  h1 {{ color: #17365d; font-size: 21pt; line-height: 1.15; margin: 0 0 12pt; }}
  h2 {{ color: #275d8c; font-size: 16pt; margin: 24pt 0 8pt; break-before: page; }}
  h3 {{ color: #17365d; font-size: 11.5pt; line-height: 1.25; margin: 15pt 0 5pt; break-after: avoid; }}
  p {{ margin: 5pt 0 8pt; }}
  strong {{ color: #17365d; }}
  table {{ width: 100%; border-collapse: collapse; margin: 9pt 0 14pt; font-size: 8.8pt; break-inside: avoid; }}
  th, td {{ border: 0.6pt solid #c9d1d9; padding: 5pt 6pt; vertical-align: top; }}
  th {{ background: #eef5fb; color: #17365d; font-weight: 700; }}
  th:not(:first-child), td:not(:first-child) {{ text-align: right; }}
  ul {{ margin: 4pt 0 9pt 18pt; padding: 0; }}
  li {{ margin: 3pt 0; break-inside: avoid; }}
  li ul {{ color: #4a5560; margin-top: 1pt; margin-bottom: 4pt; }}
  a {{ color: #275d8c; text-decoration: none; }}
  h1 + p {{ color: #4f5b66; font-size: 9.5pt; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    browser = shutil.which("google-chrome") or shutil.which("chromium")
    if browser is None:
        raise RuntimeError("google-chrome or chromium is required")

    output.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(source.read_text(encoding="utf-8"), args.title)
    with tempfile.TemporaryDirectory(prefix="libero-report-") as temp_dir:
        html_path = Path(temp_dir) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-pdf-header-footer",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={output}",
                html_path.as_uri(),
            ],
            check=True,
        )

    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
