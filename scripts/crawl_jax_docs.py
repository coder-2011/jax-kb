#!/usr/bin/env python3
"""Crawl docs.jax.dev, convert every documentation route to Markdown, and record coverage.

The acquisition strategy intentionally combines three Sphinx signals:

- the HTML link graph rooted at https://docs.jax.dev/en/latest/
- searchindex.js docnames, which enumerate rendered documentation pages
- objects.inv inventory records, which enumerate API/function/class targets

Each fetched HTML page is converted with pandoc and stored as a standalone
Markdown file under docs/pages/. The script also writes manifests that the
chunker and coverage checks consume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://docs.jax.dev/en/latest/"
ROOT_URL = "https://docs.jax.dev/"
DEFAULT_PAGES_DIR = ROOT / "docs" / "pages"
DEFAULT_HTML_DIR = ROOT / "docs" / "html"
DEFAULT_COMBINED = ROOT / "docs" / "jax-docs.md"
DEFAULT_ROUTES = ROOT / "data" / "routes.jsonl"
DEFAULT_OBJECTS = ROOT / "data" / "objects.jsonl"
DEFAULT_COVERAGE = ROOT / "data" / "coverage.json"

USER_AGENT = "jax-kb-docs-crawler/1.0 (+https://github.com/coder-2011)"
HTML_SUFFIXES = ("", ".html")
SKIP_SUFFIXES = (
    ".css",
    ".js",
    ".mjs",
    ".map",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".pdf",
    ".zip",
    ".whl",
    ".gz",
    ".txt",
    ".ipynb",
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: set[str] = set()
        self.title = ""
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag in {"a", "link"} and attr.get("href"):
            self.links.add(html.unescape(attr["href"] or ""))
        if tag in {"script", "img", "iframe"} and attr.get("src"):
            self.links.add(html.unescape(attr["src"] or ""))
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.title = " ".join("".join(self._title_parts).split())

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--pages-dir", type=Path, default=DEFAULT_PAGES_DIR)
    parser.add_argument("--html-dir", type=Path, default=DEFAULT_HTML_DIR)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--delay", type=float, default=0.04)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-pages", type=int, default=0, help="debug limit; 0 means no limit")
    parser.add_argument("--no-clean", action="store_true", help="keep existing generated docs before crawling")
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="preserve existing route records and fetch only routes missing from the coverage report",
    )
    return parser.parse_args()


def fetch(url: str, timeout: int, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 410}:
                raise
            last_error = exc
            time.sleep(min(2**attempt, 4))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def normalize_url(url: str, base_url: str) -> str | None:
    absolute = urllib.parse.urljoin(base_url, url)
    parsed = urllib.parse.urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc != "docs.jax.dev":
        return None
    path = parsed.path
    if path == "/":
        path = "/en/latest/index.html"
    if not (path.startswith("/en/latest/") or path == "/"):
        return None
    if any(path.endswith(suffix) for suffix in SKIP_SUFFIXES):
        return None
    if not path.endswith(HTML_SUFFIXES):
        return None
    if path.endswith("/"):
        path = path + "index.html"
    if not path.endswith(".html"):
        return None
    cleaned = urllib.parse.urlunparse(("https", parsed.netloc, path, "", "", ""))
    return cleaned


def url_to_route(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    if path == "/en/latest/index.html":
        return "index"
    if path.startswith("/en/latest/"):
        path = path[len("/en/latest/") :]
    return path.removesuffix(".html")


def route_to_url(route: str, base_url: str) -> str:
    if route in {"", "index"}:
        return urllib.parse.urljoin(base_url, "index.html")
    route = route.removesuffix(".html").removesuffix(".rst").removesuffix(".md").removesuffix(".ipynb")
    return urllib.parse.urljoin(base_url, route + ".html")


def slug_for_route(route: str) -> str:
    slug = route.strip("/").replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug).strip("-")
    return slug or "index"


def parse_searchindex(text: str, base_url: str) -> set[str]:
    match = re.match(r"\s*Search\.setIndex\((.*)\)\s*;?\s*$", text, flags=re.S)
    if not match:
        return set()
    payload = json.loads(match.group(1))
    routes: set[str] = set()
    for key in ("docnames", "filenames"):
        for docname in payload.get(key, []):
            route = str(docname).removesuffix(".html")
            routes.add(route_to_url(route, base_url))
    return routes


def parse_inventory(data: bytes, base_url: str) -> tuple[set[str], list[dict]]:
    lines = data.split(b"\n", 4)
    if len(lines) < 5:
        return set(), []
    text = zlib.decompress(lines[4]).decode("utf-8", errors="replace")
    routes: set[str] = set()
    objects: list[dict] = []
    pattern = re.compile(r"(?P<name>.+?)\s+(?P<domain>\S+)\s+(?P<priority>-?\d+)\s+(?P<uri>\S+)\s+(?P<dispname>.*)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        uri = match.group("uri")
        if "$" in uri:
            uri = uri.replace("$", match.group("name"))
        page_uri = uri.split("#", 1)[0]
        url = normalize_url(page_uri, base_url)
        if url:
            routes.add(url)
        objects.append(
            {
                "name": match.group("name"),
                "domain": match.group("domain"),
                "priority": int(match.group("priority")),
                "uri": uri,
                "url": urllib.parse.urljoin(base_url, uri),
                "route": url_to_route(url) if url else "",
                "display_name": "" if match.group("dispname") == "-" else match.group("dispname"),
            }
        )
    return routes, objects


def convert_html_to_markdown(html_path: Path, output_path: Path, source_url: str) -> None:
    command = [
        "pandoc",
        "--from",
        "html",
        "--to",
        "gfm-raw_html",
        "--wrap=none",
        "--metadata",
        f"source-url={source_url}",
        str(html_path),
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def process_page(url: str, args: argparse.Namespace) -> dict:
    route = url_to_route(url)
    slug = slug_for_route(route)
    html_path = args.html_dir / f"{slug}.html"
    markdown_path = args.pages_dir / f"{slug}.md"
    body = fetch(url, args.timeout)
    html_text = body.decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html_text)
    html_path.write_text(html_text, encoding="utf-8")
    convert_html_to_markdown(html_path, markdown_path, url)
    markdown_text = markdown_path.read_text(encoding="utf-8", errors="replace")
    return {
        "route": route,
        "url": url,
        "title": parser.title.replace(" — JAX documentation", "").strip(),
        "html_path": str(html_path.relative_to(ROOT)),
        "markdown_path": str(markdown_path.relative_to(ROOT)),
        "bytes_html": len(body),
        "bytes_markdown": len(markdown_text.encode("utf-8")),
        "sha256_html": hashlib.sha256(body).hexdigest(),
        "links": sorted(parser.links),
    }


def clean_generated(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_combined(path: Path, route_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# JAX Documentation Knowledge Base\n\n")
        handle.write(f"Source root: {BASE_URL}\n\n")
        handle.write("This generated file concatenates every Markdown page captured by `scripts/crawl_jax_docs.py`.\n\n")
        for row in route_rows:
            page_path = ROOT / row["markdown_path"]
            text = page_path.read_text(encoding="utf-8", errors="replace").strip()
            handle.write("\n\n---\n\n")
            handle.write(f"# Route: {row['route']}\n\n")
            handle.write(f"- Source URL: {row['url']}\n")
            handle.write(f"- Title: {row['title']}\n")
            handle.write(f"- Markdown: {row['markdown_path']}\n\n")
            handle.write(text)
            handle.write("\n")


def main() -> None:
    args = parse_args()
    if not shutil.which("pandoc"):
        raise SystemExit("pandoc is required to convert HTML documentation to Markdown")

    base_url = args.base_url.rstrip("/") + "/"
    if args.retry_missing:
        args.no_clean = True
    if not args.no_clean:
        clean_generated([args.pages_dir, args.html_dir, args.combined, args.routes, args.objects, args.coverage])
    args.pages_dir.mkdir(parents=True, exist_ok=True)
    args.html_dir.mkdir(parents=True, exist_ok=True)
    args.routes.parent.mkdir(parents=True, exist_ok=True)

    discovered: set[str] = {route_to_url("index", base_url)}
    searchindex_url = urllib.parse.urljoin(base_url, "searchindex.js")
    objects_url = urllib.parse.urljoin(base_url, "objects.inv")

    searchindex_routes = parse_searchindex(fetch(searchindex_url, args.timeout).decode("utf-8", errors="replace"), base_url)
    inventory_routes, objects = parse_inventory(fetch(objects_url, args.timeout), base_url)
    discovered.update(searchindex_routes)
    discovered.update(inventory_routes)

    seen: set[str] = set()
    route_rows: list[dict] = []
    failed: list[dict] = []
    skipped_not_found: list[dict] = []
    searchindex_expected = {url_to_route(url) for url in searchindex_routes}
    inventory_expected = {url_to_route(url) for url in inventory_routes}
    retry_existing_routes: set[str] = set()

    if args.retry_missing:
        existing_rows = read_jsonl(args.routes)
        existing_by_route = {row["route"]: row for row in existing_rows}
        route_rows = list(existing_by_route.values())
        fetched_routes = set(existing_by_route)
        retry_existing_routes = set(fetched_routes)
        previous_coverage = json.loads(args.coverage.read_text(encoding="utf-8")) if args.coverage.exists() else {}
        retry_urls = {
            entry["url"]
            for entry in previous_coverage.get("failed", [])
            if entry.get("url")
        }
        retry_urls.update(route_to_url(route, base_url) for route in previous_coverage.get("missing_searchindex_routes", []))
        retry_urls.update(route_to_url(route, base_url) for route in previous_coverage.get("missing_inventory_routes", []))
        retry_urls.update(url for url in discovered if url_to_route(url) not in fetched_routes)
        discovered = retry_urls

    workers = max(args.workers, 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            if args.max_pages:
                remaining = args.max_pages - len(route_rows)
                if remaining <= 0:
                    break
            else:
                remaining = 0
            pending = sorted(discovered - seen)
            if remaining:
                pending = pending[:remaining]
            if not pending:
                break
            for url in pending:
                seen.add(url)

            futures = {executor.submit(process_page, url, args): url for url in pending}
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                route = url_to_route(url)
                try:
                    row = future.result()
                    links = row.pop("links")
                    for link in links:
                        normalized = normalize_url(link, url)
                        if args.retry_missing and normalized and url_to_route(normalized) in retry_existing_routes:
                            continue
                        if normalized and normalized not in seen:
                            discovered.add(normalized)
                    row["from_searchindex"] = route in searchindex_expected
                    row["from_inventory"] = route in inventory_expected
                    route_rows.append(row)
                    print(f"[{len(route_rows):04d}] {route}", flush=True)
                    if args.delay:
                        time.sleep(args.delay)
                except Exception as exc:
                    if (
                        isinstance(exc, urllib.error.HTTPError)
                        and exc.code in {404, 410}
                        and route not in searchindex_expected
                        and route not in inventory_expected
                    ):
                        skipped_not_found.append({"route": route, "url": url, "error": str(exc)})
                        print(f"SKIPPED {url}: {exc}", file=sys.stderr, flush=True)
                    else:
                        failed.append({"route": route, "url": url, "error": str(exc)})
                        print(f"FAILED {url}: {exc}", file=sys.stderr, flush=True)

    route_rows = list({row["route"]: row for row in route_rows}.values())
    route_rows.sort(key=lambda row: row["route"])
    objects.sort(key=lambda row: (row["route"], row["name"]))
    fetched_routes = {row["route"] for row in route_rows}
    coverage = {
        "base_url": base_url,
        "searchindex_url": searchindex_url,
        "objects_url": objects_url,
        "routes_fetched": len(route_rows),
        "routes_failed": len(failed),
        "objects_total": len(objects),
        "searchindex_routes": len(searchindex_expected),
        "inventory_routes": len(inventory_expected),
        "missing_searchindex_routes": sorted(searchindex_expected - fetched_routes),
        "missing_inventory_routes": sorted(inventory_expected - fetched_routes),
        "failed": failed,
        "skipped_not_found": skipped_not_found,
    }
    write_jsonl(args.routes, route_rows)
    write_jsonl(args.objects, objects)
    write_combined(args.combined, route_rows)
    args.coverage.write_text(json.dumps(coverage, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: coverage[k] for k in ("routes_fetched", "routes_failed", "objects_total", "searchindex_routes", "inventory_routes")}, indent=2))


if __name__ == "__main__":
    main()
