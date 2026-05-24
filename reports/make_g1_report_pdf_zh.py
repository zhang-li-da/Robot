from __future__ import annotations

import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


REPORT_MD = Path(__file__).with_name("g1_knee_climb_50cm_report_zh.md")
REPORT_PDF = Path(__file__).with_name("g1_knee_climb_50cm_report_zh.pdf")
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def display_width(char: str) -> int:
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def wrap_display(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    lines: list[str] = []
    current = ""
    current_width = 0
    indent = "  " if line.startswith(("- ", "  ")) else ""
    for char in line:
        char_width = display_width(char)
        if current and current_width + char_width > width:
            lines.append(current.rstrip())
            current = indent + char
            current_width = sum(display_width(c) for c in current)
        else:
            current += char
            current_width += char_width
    if current:
        lines.append(current.rstrip())
    return lines


def build_pages(text: str, width: int = 72, lines_per_page: int = 42) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    in_code = False
    for raw_line in text.splitlines():
        if raw_line.startswith("```"):
            in_code = not in_code
            wrapped = [raw_line]
        elif in_code:
            wrapped = wrap_display(raw_line, 82)
        else:
            wrapped = wrap_display(raw_line, width)
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
    font_regular = FontProperties(fname=FONT_REGULAR)
    font_bold = FontProperties(fname=FONT_BOLD)
    font_mono = FontProperties(fname=FONT_REGULAR)

    with PdfPages(REPORT_PDF) as pdf:
        for page_index, lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.955
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                    size = 7.2
                    prop = font_mono
                elif line.startswith("# "):
                    size = 15
                    prop = font_bold
                elif line.startswith("## "):
                    size = 11.3
                    prop = font_bold
                elif in_code:
                    size = 6.8
                    prop = font_mono
                else:
                    size = 8.5
                    prop = font_regular
                ax.text(0.065, y, line, fontproperties=prop, fontsize=size, va="top")
                y -= 0.018 if size <= 8.5 else 0.025
            ax.text(
                0.5,
                0.025,
                f"Unitree G1 50cm 膝爬越障实验报告 - 第 {page_index}/{len(pages)} 页",
                ha="center",
                fontproperties=font_regular,
                fontsize=7,
                color="0.35",
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(REPORT_PDF)


if __name__ == "__main__":
    main()
