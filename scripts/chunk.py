#!/usr/bin/env python3
"""Chunk crawled JAX documentation Markdown into route-aware search records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTES = ROOT / "data" / "routes.jsonl"
DEFAULT_OBJECTS = ROOT / "data" / "objects.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_SECTIONS = ROOT / "data" / "sections.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--words", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def markdown_headings(text: str) -> list[dict]:
    headings: list[dict] = []
    stack: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = re.sub(r"\s+#$", "", match.group(2)).strip()
        title = re.sub(r"\[¶\]\(.*?\)", "", title).strip()
        if not title:
            continue
        stack = stack[: level - 1]
        stack.append(title)
        headings.append({"level": level, "title": title, "line": line_number, "heading_path": list(stack)})
    return headings


def section_for_line(headings: list[dict], line_number: int, fallback: str) -> tuple[str, list[str]]:
    active: list[str] = [fallback]
    for heading in headings:
        if heading["line"] <= line_number:
            active = heading["heading_path"]
        else:
            break
    return " > ".join(active), active


def chunk_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(words):
        chunks.append(words[start : start + size])
        if start + size >= len(words):
            break
        start += step
    return chunks


def line_offsets(text: str) -> list[int]:
    offsets = []
    position = 0
    for line in text.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    return offsets


def line_for_offset(offsets: list[int], offset: int) -> int:
    line = 1
    for index, line_offset in enumerate(offsets, start=1):
        if line_offset > offset:
            return max(line - 1, 1)
        line = index
    return max(line, 1)


def make_chunks(route_rows: list[dict], words_per_chunk: int, overlap: int) -> tuple[list[dict], list[dict]]:
    chunks: list[dict] = []
    sections: list[dict] = []

    for route_row in route_rows:
        markdown_path = ROOT / route_row["markdown_path"]
        raw = markdown_path.read_text(encoding="utf-8", errors="replace")
        text = normalize_text(raw)
        headings = markdown_headings(text)
        offsets = line_offsets(text)
        route = route_row["route"]
        title = route_row.get("title") or route
        if not headings:
            headings = [{"level": 1, "title": title, "line": 1, "heading_path": [title]}]

        for heading in headings:
            sections.append(
                {
                    "id": f"section-{len(sections):06d}",
                    "title": heading["title"],
                    "level": heading["level"],
                    "route": route,
                    "url": route_row["url"],
                    "source": route_row["markdown_path"],
                    "heading_path": heading["heading_path"],
                    "search_text": f"{route}\n{route_row['url']}\n" + " > ".join(heading["heading_path"]),
                }
            )

        words_with_offsets = [(match.group(0), match.start()) for match in re.finditer(r"\S+", text)]
        words = [word for word, _ in words_with_offsets]
        for chunk_index, word_chunk in enumerate(chunk_words(words, words_per_chunk, overlap)):
            start_word = chunk_index * max(words_per_chunk - overlap, 1)
            offset = words_with_offsets[min(start_word, len(words_with_offsets) - 1)][1] if words_with_offsets else 0
            line_number = line_for_offset(offsets, offset)
            heading, heading_path = section_for_line(headings, line_number, title)
            chunk_text = " ".join(word_chunk)
            chunks.append(
                {
                    "id": f"route-{len(chunks):06d}",
                    "source": route_row["markdown_path"],
                    "url": route_row["url"],
                    "route": route,
                    "title": title,
                    "heading": heading,
                    "heading_path": heading_path,
                    "line": line_number,
                    "text": chunk_text,
                    "search_text": f"{route}\n{route_row['url']}\n{heading}\n{chunk_text}",
                }
            )

    return chunks, sections


def make_object_sections(objects: list[dict]) -> list[dict]:
    rows = []
    for obj in objects:
        rows.append(
            {
                "id": f"object-{len(rows):06d}",
                "title": obj["name"],
                "level": 1,
                "route": obj.get("route", ""),
                "url": obj["url"],
                "source": "data/objects.jsonl",
                "heading_path": [obj["name"]],
                "object_domain": obj["domain"],
                "search_text": f"{obj['name']}\n{obj['domain']}\n{obj['url']}\n{obj.get('route', '')}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    route_rows = read_jsonl(args.routes)
    objects = read_jsonl(args.objects) if args.objects.exists() else []
    chunks, sections = make_chunks(route_rows, args.words, args.overlap)
    sections.extend(make_object_sections(objects))
    write_jsonl(args.output, chunks)
    write_jsonl(args.sections, sections)
    print(f"wrote {len(chunks)} chunks to {args.output}")
    print(f"wrote {len(sections)} section/object records to {args.sections}")


if __name__ == "__main__":
    main()
