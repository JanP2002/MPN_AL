"""Small self-contained HTML reports (Polish): tables, embedded pictures, notes written by hand, and a PDF via a browser."""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

STYLE = """
@page { size: A4 landscape; margin: 11mm; }
:root { --fg:#1d1d1f; --mute:#5f6368; --line:#d7d9dd; --mark:#e8f1fb; --warn:#fff4e0; }
* { box-sizing: border-box; }
body { font: 14px/1.5 "Segoe UI", system-ui, sans-serif; color: var(--fg); background:#fff; max-width: 1180px; margin: 0 auto; padding: 0 18px 40px; }
h1 { font-size: 26px; margin: 18px 0 4px; } h2 { font-size: 19px; border-bottom: 2px solid var(--line); padding-bottom: 4px; margin-top: 30px; }
h3 { font-size: 15px; margin: 18px 0 6px; } .mute { color: var(--mute); }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 12.5px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f5f6f8; } td.n { text-align: right; white-space: nowrap; } th.n { text-align: right; }
tr.mark td, td.mark { background: var(--mark); font-weight: 600; }
.box { background: var(--mark); border-radius: 8px; padding: 10px 16px; margin: 8px 0; }
.warn { background: var(--warn); border-radius: 8px; padding: 10px 16px; margin: 8px 0; }
.note { border-left: 3px solid var(--line); padding: 2px 0 2px 12px; margin: 8px 0; }
figure { margin: 10px 0 18px; } img { width: 100%; height: auto; display: block; margin: 0 auto; }
img.natural { width: auto; max-width: 100%; }
figcaption { font-size: 12px; color: var(--mute); margin-top: 4px; }
.page { break-before: page; } table, figure, .box, .warn { break-inside: avoid; } .keep { break-inside: avoid; }
li { margin: 3px 0; }
@media print { body { max-width: none; padding: 0; font-size: 11.5px; } img { max-height: 158mm; width: auto; max-width: 100%; }
  h2, h3 { break-after: avoid; } th, td { padding: 2px 6px; } h2 { margin-top: 14px; } }
"""

_NUMBER = re.compile(r"[−+-]?[0-9][0-9 ,.%±/()–−+-]*")


def num(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",").replace("-", "−")


def percent(share: float) -> str:
    return "–" if share is None or not np.isfinite(share) else f"{100 * share:.0f}%"


def table(header: Sequence, rows: Sequence[Sequence], *, mark_rows: Sequence[int] = (), mark_cells: Sequence[tuple[int, int]] = ()) -> str:
    marked, cells_marked = set(mark_rows), set(mark_cells)
    numeric_column = [bool(rows) and all(_NUMBER.fullmatch(str(r[c])) or str(r[c]) == "–" for r in rows) for c in range(len(header))]
    head = "".join(f"<th{' class=n' if numeric_column[c] else ''}>{html.escape(str(h))}</th>" for c, h in enumerate(header))
    body = []
    for i, row in enumerate(rows):
        cells = []
        for c, value in enumerate(row):
            classes = [name for name, on in (("n", numeric_column[c]), ("mark", (i, c) in cells_marked)) if on]
            attribute = f" class='{' '.join(classes)}'" if classes else ""
            cells.append(f"<td{attribute}>{html.escape(str(value))}</td>")
        body.append(f"<tr{' class=mark' if i in marked else ''}>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def picture(image: str | Path | np.ndarray, caption: str, *, width: int = 2400, quality: int = 88, css: str = "", lossless: bool = False) -> str:
    """An embedded figure from a file or an RGB array, scaled down to ``width``; ``lossless`` keeps PNG (charts)."""
    if isinstance(image, (str, Path)):
        data = np.frombuffer(Path(image).read_bytes(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read the image {image}")
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] > width:
        bgr = cv2.resize(bgr, (width, round(bgr.shape[0] * width / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    kind, options = ("png", []) if lossless else ("jpeg", [cv2.IMWRITE_JPEG_QUALITY, quality])
    ok, buffer = cv2.imencode(".png" if lossless else ".jpg", bgr, options)
    if not ok:
        raise RuntimeError("image encoding failed")
    source = f"data:image/{kind};base64,{base64.b64encode(buffer.tobytes()).decode()}"
    return (f"<figure class='{css}'><img alt='{html.escape(caption)}' src='{source}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def read_notes(path: str | Path | None) -> dict[str, str]:
    """Hand-written commentary: sections headed ``## key``; lines starting ``- `` become a list, ``**text**`` bold."""
    if path is None or not Path(path).exists():
        return {}
    sections: dict[str, list[str]] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {key: _to_html(lines) for key, lines in sections.items()}


def _to_html(lines: Sequence[str]) -> str:
    out, items, paragraph = [], [], []

    def inline(text: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text, quote=False))

    def flush():
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in lines:
        text = line.strip()
        if not text:
            flush()
        elif text.startswith("- "):
            if paragraph:
                flush()
            items.append(text[2:])
        elif items:
            items[-1] += " " + text
        else:
            paragraph.append(text)
    flush()
    return "".join(out)


def page(title: str, body: str, *, extra_style: str = "") -> str:
    return (f"<!doctype html><html lang='pl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}{extra_style}</style></head><body>{body}</body></html>")


def find_browser() -> str | None:
    candidates = [shutil.which(n) for n in ("msedge", "chrome", "chromium", "google-chrome")]
    candidates += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    return next((c for c in candidates if c and Path(c).exists()), None)


def to_pdf(source: Path, target: Path, *, timeout: int = 300) -> bool:
    """Prints an HTML file to PDF with a headless Edge or Chrome; False when there is none or it fails."""
    browser = find_browser()
    if browser is None:
        return False
    target = Path(target).resolve()
    target.unlink(missing_ok=True)
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={target}", Path(source).resolve().as_uri()],
                   check=False, timeout=timeout, capture_output=True)
    return target.exists() and target.stat().st_size > 0
