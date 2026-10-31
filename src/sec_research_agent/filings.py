"""Turn a 10-K / 10-Q HTML document into clean text and pull out the narrative sections."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString

from .models import Section

BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]

# item code -> section, per form type
SECTION_ITEMS: dict[str, dict[str, Section]] = {
    "10-K": {"1": "business", "1A": "risk_factors", "7": "mdna"},
    "10-Q": {"2": "mdna", "1A": "risk_factors"},
}

HEADING = re.compile(r"^\s*item\s+(\d{1,2}[a-d]?)\b[\s\.:\-–—]*(.*)$", re.IGNORECASE | re.MULTILINE)
MIN_SECTION_CHARS = 500
MAX_HEADING_CHARS = 120


def _is_numeric_table(table) -> bool:
    text = table.get_text(" ", strip=True)
    if not text:
        return True
    digits = sum(ch.isdigit() for ch in text)
    return digits / len(text) > 0.15 or text.count("$") >= 3


def html_to_text(html: str) -> str:
    """Readable text with one line per block. Financial tables are dropped: numbers come
    from XBRL. Layout tables stay, since many filings put item headings inside them."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()
    for table in soup.find_all("table"):
        if not table.decomposed and _is_numeric_table(table):
            table.decompose()
    for tag in soup.find_all(["td", "th"]):
        tag.append(NavigableString(" "))
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append(NavigableString("\n"))
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def _base_form(form: str) -> str:
    return form.split("/")[0]


def split_sections(text: str, form: str) -> dict[Section, str]:
    """Find each wanted item by its heading. The table of contents also lists every item,
    so for each item keep the longest span between its heading and the next item heading."""
    items = SECTION_ITEMS.get(_base_form(form), {})
    headings = [
        (m.start(), m.end(), m.group(1).upper())
        for m in HEADING.finditer(text)
        if len(m.group(0).strip()) <= MAX_HEADING_CHARS
    ]
    sections: dict[Section, str] = {}
    for code, section in items.items():
        best = ""
        for i, (start, _end, found) in enumerate(headings):
            if found != code:
                continue
            stop = len(text)
            for later_start, _, later_code in headings[i + 1 :]:
                if later_code != code:
                    stop = later_start
                    break
            body = text[start:stop].strip()
            if len(body) > len(best):
                best = body
        best = clean_section(best)
        if len(best) >= MIN_SECTION_CHARS:
            sections[section] = best
    return sections


PAGE_JUNK = re.compile(
    r"^(\d{1,3}"                                   # bare page number
    r"|table of contents"                          # running link back to the TOC
    r"|\(continued\)"
    r"|part\s+[ivx]+"                              # running PART I / PART II headers
    r"|item\s+\d{1,2}[a-d]?\.?"                    # running item headers (kept on the first line)
    r"|.{0,80}\|\s*\d{4}\s+form\s+10-[kq]\s*\|\s*\d{1,3}"  # "Apple Inc. | 2025 Form 10-K | 11"
    r")$",
    re.IGNORECASE,
)


def clean_section(body: str) -> str:
    """Drop page furniture (page numbers, running headers, footers) that would pollute chunks."""
    lines = body.splitlines()
    if not lines:
        return body
    kept = [lines[0]] + [line for line in lines[1:] if not PAGE_JUNK.match(line.strip())]
    return "\n".join(kept).strip()
