"""File I/O for the wiki — class-based, no module-level globals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def _strip_code_fence(content: str) -> str:
    """Remove wrapping code fences that LLMs sometimes add around the whole page."""
    content = content.strip()
    # Match opening fence: ```yaml, ```markdown, ```md, or plain ```
    content = re.sub(r"^```[a-z]*\n?", "", content)
    # Match closing fence
    content = re.sub(r"\n?```\s*$", "", content)
    return content.strip()


class WikiStorage:
    """Manages wiki page and raw-source file I/O under a configurable root."""

    def __init__(self, wiki_root: Path) -> None:
        self.wiki_root = Path(wiki_root)
        self._wiki_dir = self.wiki_root / "wiki"
        self._raw_dir = self.wiki_root / "raw"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _safe_path(self, base: Path, name: str) -> Path:
        """Resolve name to a path under base, blocking directory traversal.
        Forward slashes are allowed for subfolder paths (e.g. 'rag/technique-chunking').
        Only '..' traversal is blocked.
        """
        if ".." in Path(name).parts or "\\" in name:
            raise ValueError(f"Path traversal detected: {name!r}")
        resolved = (base / name).resolve()
        if not str(resolved).startswith(str(base.resolve())):
            raise ValueError(f"Path traversal detected: {name!r}")
        return resolved

    def _wiki_path(self, name: str) -> Path:
        if not name.endswith(".md"):
            name = name + ".md"
        return self._safe_path(self._wiki_dir, name)

    def _raw_path(self, name: str) -> Path:
        if not name.endswith(".md"):
            name = name + ".md"
        return self._safe_path(self._raw_dir, name)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_page(self, name: str) -> Optional[str]:
        """Return page content, or None if the page does not exist."""
        path = self._wiki_path(name)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def read_raw(self, name: str) -> Optional[str]:
        """Return raw source content, or None if the file does not exist."""
        path = self._raw_path(name)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_page(self, name: str, content: str) -> Path:
        """Write (or overwrite) a wiki page under wiki_root/wiki/."""
        path = self._wiki_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_strip_code_fence(content), encoding="utf-8")
        return path

    def save_raw(self, name: str, content: str) -> Path:
        """Save raw source content under wiki_root/raw/."""
        path = self._raw_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_pages(self) -> list[str]:
        """Return a sorted list of page names (without .md extension)."""
        if not self._wiki_dir.exists():
            return []
        return sorted(
            p.stem
            for p in self._wiki_dir.rglob("*.md")
            if p.is_file()
        )

    # ------------------------------------------------------------------
    # New-content log
    # ------------------------------------------------------------------

    def append_new_content(self, source_title: str, source_url: str, pages: list[dict]) -> None:
        """Append a row per written page to new-content.md (creates file if missing)."""
        from datetime import date
        log_path = self.wiki_root / "new-content.md"

        if not log_path.exists():
            log_path.write_text(
                "# Recent Wiki Updates\n\n"
                "| Date | Page | Action | Source |\n"
                "|---|---|---|---|\n",
                encoding="utf-8",
            )

        today = date.today().isoformat()
        lines = []
        for p in pages:
            # Use filename only — Obsidian resolves wikilinks by filename across the vault
            filename = p["page"].split("/")[-1]
            page_link = f"[[{filename}|{p['title']}]]"
            source_link = f"[{source_title}]({source_url})" if source_url else source_title
            lines.append(f"| {today} | {page_link} | {p['action']} | {source_link} |\n")

        with open(log_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Full-text search over wiki pages.

        Returns a list of ``{name, score, snippet}`` dicts ranked by
        descending match count, capped at *max_results*.
        """
        if not self._wiki_dir.exists():
            return []

        terms = [t.lower() for t in query.split() if t]
        if not terms:
            return []

        results: list[tuple[int, str, str]] = []

        for page_path in self._wiki_dir.rglob("*.md"):
            text = page_path.read_text(encoding="utf-8")
            lower = text.lower()
            score = sum(lower.count(t) for t in terms)
            if score == 0:
                continue
            first_term = terms[0]
            idx = lower.find(first_term)
            start = max(0, idx - 60)
            snippet = text[start : start + 200].replace("\n", " ").strip()
            name = page_path.stem
            results.append((score, name, snippet))

        results.sort(reverse=True)
        return [
            {"name": name, "score": score, "snippet": snippet}
            for score, name, snippet in results[:max_results]
        ]
