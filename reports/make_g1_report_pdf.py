from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_MD = Path(__file__).with_name("g1_knee_climb_50cm_report.md")
REPORT_PDF = Path(__file__).with_name("g1_knee_climb_50cm_report.pdf")


def _wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    prefix = ""
    stripped = line.lstrip()
    if line.startswith("- "):
        prefix = "  "
    elif line.startswith("  "):
        prefix = "  "
    return textwrap.wrap(
        line,
        width=width,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [line]


def build_pages(text: str, width: int = 96, lines_per_page: int = 56) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    in_code = False
    for raw_line in text.splitlines():
        if raw_line.startswith("```"):
            in_code = not in_code
            wrapped = [raw_line]
        elif in_code:
            wrapped = [raw_line[i : i + width] for i in range(0, len(raw_line), width)] or [""]
        else:
            wrapped = _wrap_line(raw_line, width)
        for line in wrapped:
            if len(current) >= lines_per_page:
                pages.append(current)
                current = []
            current.append(line)
    if current:
        pages.append(current)
    return pages


def main() -> None:
    text = REPORT_MD.read_text(encoding="utf-8")
    pages = build_pages(text)
    with PdfPages(REPORT_PDF) as pdf:
        for page_index, lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.965
            for line in lines:
                if line.startswith("# "):
                    size = 14
                    weight = "bold"
                elif line.startswith("## "):
                    size = 11.5
                    weight = "bold"
                else:
                    size = 8.1
                    weight = "normal"
                ax.text(
                    0.06,
                    y,
                    line,
                    family="DejaVu Sans Mono",
                    fontsize=size,
                    fontweight=weight,
                    va="top",
                )
                y -= 0.0165 if size <= 8.1 else 0.021
            ax.text(
                0.5,
                0.025,
                f"Unitree G1 50 cm Knee-Climb Report - page {page_index}/{len(pages)}",
                ha="center",
                family="DejaVu Sans",
                fontsize=7,
                color="0.35",
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(REPORT_PDF)


if __name__ == "__main__":
    main()
