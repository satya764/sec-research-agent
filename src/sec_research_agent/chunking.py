"""Split section text into overlapping chunks: paragraphs first, then sentences, then hard cuts."""

from __future__ import annotations

import re


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend([p] if len(p) <= max_chars else _split_long(p, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current}\n{piece}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            if tail and len(tail) + 1 + len(piece) <= max_chars:
                current = f"{tail}\n{piece}"
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks
