from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable

from .fetcher import ArticleFetcher
from .storage import WikiStorage


class WikiDocumenter:
    """Document tech articles and research topics into an Obsidian wiki vault."""

    def __init__(self, wiki_root: Path, llm_fn: Callable[[str], Awaitable[str]]):
        self.storage = WikiStorage(wiki_root)
        self.fetcher = ArticleFetcher()
        self.llm_fn = llm_fn

    async def document(self, url: str, title: str) -> dict:
        """Fetch a URL, save raw content, and write/enhance a wiki page."""
        content = await self.fetcher.fetch(url)
        slug = _slugify(title)

        self.storage.save_raw(slug, content)

        existing = self.storage.read_page(slug)
        prompt = _build_document_prompt(title, url, content, existing)
        wiki_markdown = await self.llm_fn(prompt)

        path = self.storage.write_page(slug, wiki_markdown)
        action = "enhanced" if existing else "created"
        return {"page": slug, "action": action, "path": str(path)}

    async def research(self, topic: str) -> dict:
        """Research a topic using LLM search grounding and write/enhance a wiki page."""
        slug = _slugify(topic)
        existing = self.storage.read_page(slug)
        prompt = _build_research_prompt(topic, existing)
        wiki_markdown = await self.llm_fn(prompt)
        path = self.storage.write_page(slug, wiki_markdown)
        action = "enhanced" if existing else "created"
        return {"page": slug, "action": action, "path": str(path)}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SCHEMA_REMINDER = """\
Every wiki page MUST have this YAML frontmatter:
---
title: "Human Readable Title"
type: tool | technique | concept | comparison | source | overview
tags: [relevant, tags]
sources: []
updated: "{today}"
---

Close the page with a ## Sources section listing the article URL and today's date.
"""

_TODAY = date.today().isoformat()


def _build_document_prompt(title: str, url: str, content: str, existing: str | None) -> str:
    existing_section = ""
    if existing:
        existing_section = f"""
## Existing wiki page (enhance and update this — do not lose information):

{existing}

---
"""
    return f"""\
You are a technical wiki writer. Write a structured wiki page based on the article below.

{_SCHEMA_REMINDER.format(today=_TODAY)}

Rules:
- Use ONLY the article content as your source — no training knowledge
- If the article covers a tool, name the page type "tool". If a concept, "concept". If a technique, "technique".
- Be precise and factual. Include version numbers, dates, and specifics mentioned in the article.
- Cross-link related concepts using [[wikilink]] notation.
{existing_section}
## Article title: {title}
## Article URL: {url}

## Article content:

{content[:8000]}

Now write the complete wiki page markdown (frontmatter + body + Sources section):"""


def _build_research_prompt(topic: str, existing: str | None) -> str:
    existing_section = ""
    if existing:
        existing_section = f"""
## Existing wiki page (enhance and update this — do not lose information):

{existing}

---
"""
    return f"""\
You are a technical wiki writer. Research the topic "{topic}" using your search capabilities \
and write a comprehensive, current wiki page.

{_SCHEMA_REMINDER.format(today=_TODAY)}

Rules:
- Use your web search / grounding capabilities to find current information (2025-2026)
- Include the current version, recent changes, and what practitioners recommend today
- Be precise and factual. Include version numbers, dates, benchmark numbers where available.
- Cross-link related concepts using [[wikilink]] notation.
- Close with a ## Sources section with real URLs and retrieval dates.
- If you cannot find current information on a specific aspect, add:
  > ⚠️ Data gap: No recent information found on [aspect].
{existing_section}
## Topic: {topic}

Write the complete wiki page markdown (frontmatter + body + Sources section):"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
