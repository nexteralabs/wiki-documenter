from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable

from .fetcher import ArticleFetcher
from .storage import WikiStorage

_TODAY = date.today().isoformat()


class WikiDocumenter:
    """Document tech articles and research topics into an Obsidian wiki vault."""

    def __init__(self, wiki_root: Path, llm_fn: Callable[[str], Awaitable[str]]):
        self.storage = WikiStorage(wiki_root)
        self.fetcher = ArticleFetcher()
        self.llm_fn = llm_fn

    async def document(self, url: str, title: str) -> dict:
        """
        Fetch a URL, identify all distinct topics, and write/enhance one wiki
        page per topic.  Returns a dict with the list of pages written.
        """
        content = await self.fetcher.fetch(url)
        raw_slug = _slugify(title)
        self.storage.save_raw(raw_slug, content)

        # Step 1 — identify topics covered by this article
        topics = await self._identify_topics(title, content)

        # Step 2 — write/enhance one wiki page per topic
        pages = []
        for topic in topics:
            existing = self.storage.read_page(topic["slug"])
            prompt = _build_topic_page_prompt(
                topic_title=topic["title"],
                topic_slug=topic["slug"],
                topic_type=topic["type"],
                article_title=title,
                article_url=url,
                article_content=content,
                existing=existing,
            )
            wiki_markdown = await self.llm_fn(prompt)
            path = self.storage.write_page(topic["slug"], wiki_markdown)
            action = "enhanced" if existing else "created"
            pages.append({
                "page": topic["slug"],
                "title": topic["title"],
                "type": topic["type"],
                "action": action,
                "path": str(path),
            })

        # Step 3 — update new-content log
        self.storage.append_new_content(title, url, pages)

        return {"pages": pages, "raw": raw_slug, "source_title": title, "source_url": url}

    async def research(self, topic: str) -> dict:
        """Research a topic using LLM search grounding and write/enhance a wiki page."""
        slug = _slugify(topic)
        existing = self.storage.read_page(slug)
        prompt = _build_research_prompt(topic, existing)
        wiki_markdown = await self.llm_fn(prompt)
        path = self.storage.write_page(slug, wiki_markdown)
        action = "enhanced" if existing else "created"

        pages = [{"page": slug, "title": topic, "type": "concept", "action": action, "path": str(path)}]
        self.storage.append_new_content(f"research: {topic}", "", pages)

        return {"pages": pages, "source_title": topic, "source_url": ""}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _identify_topics(self, title: str, content: str) -> list[dict]:
        """Ask the LLM to extract distinct topics from the article."""
        prompt = _build_topics_prompt(title, content)
        raw = await self.llm_fn(prompt)
        return _parse_topics_json(raw)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_PAGE_FORMAT = """\
Every wiki page MUST follow this exact structure:

```
---
title: "Human Readable Title"
type: tool | technique | concept | comparison | overview
tags: [relevant, tags]
sources: []
updated: "{today}"
---

## TL;DR
[2–4 sentences. The single most useful thing to know about this topic — the takeaway a busy engineer needs in 30 seconds.]

## Details
[Comprehensive technical content. Subsections as needed. Be precise: include version numbers, specifics, trade-offs, and code examples if relevant. Cross-link related topics using [[wikilink]] notation.]

## Sources
- [Title](url) — retrieved {today}
```
"""


def _build_topics_prompt(title: str, content: str) -> str:
    return f"""\
You are analyzing a tech article to identify all distinct topics it covers.

Article title: {title}

Article content (first 6000 chars):
{content[:6000]}

---

Identify every distinct topic, technique, tool, or concept this article covers.
Each topic should be granular enough to stand alone as a wiki page.

Return ONLY a JSON array, no other text:
[
  {{"slug": "technique-prompt-caching", "title": "Prompt Caching", "type": "technique"}},
  {{"slug": "technique-semantic-caching", "title": "Semantic Caching", "type": "technique"}},
  ...
]

Rules:
- slug: lowercase, hyphen-separated, prefixed with type (tool-, technique-, concept-, compare-)
- type: tool | technique | concept | comparison | overview
- 3 to 8 topics max — only meaningful distinct subjects, not every passing mention
- If a topic already has an obvious wiki slug (e.g. "technique-rag-variants"), use it
"""


def _build_topic_page_prompt(
    topic_title: str,
    topic_slug: str,
    topic_type: str,
    article_title: str,
    article_url: str,
    article_content: str,
    existing: str | None,
) -> str:
    existing_section = ""
    if existing:
        existing_section = f"""
## Existing wiki page for this topic — preserve all existing info, add/update with new:

{existing}

---
"""
    return f"""\
You are a technical wiki writer. Write a focused wiki page about ONE specific topic.

{_PAGE_FORMAT.format(today=_TODAY)}

Topic to document: {topic_title} (type: {topic_type}, slug: {topic_slug})
Source article: {article_title}
Source URL: {article_url}
{existing_section}
Extract ONLY what the article says about "{topic_title}". Do not pad with unrelated content.
Use only the article as your source — no training knowledge.
{f"Cross-link related topics using [[wikilink]] notation." if not existing else ""}

Article content:
{article_content[:8000]}

Write the complete wiki page now:"""


def _build_research_prompt(topic: str, existing: str | None) -> str:
    existing_section = ""
    if existing:
        existing_section = f"""
## Existing wiki page — preserve all, add/update with current research:

{existing}

---
"""
    return f"""\
You are a technical wiki writer. Research "{topic}" and write a comprehensive wiki page.

{_PAGE_FORMAT.format(today=_TODAY)}

Rules:
- Use your web search / grounding capabilities to find current info (2025–2026)
- Include current versions, recent changes, what practitioners recommend today
- Be precise: version numbers, dates, benchmark numbers where available
- Cross-link related topics using [[wikilink]] notation
- If current info is unavailable on a specific aspect, add:
  > ⚠️ Data gap: No recent information found on [aspect]
{existing_section}
Topic: {topic}

Write the complete wiki page now:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_topics_json(raw: str) -> list[dict]:
    """Extract JSON array from LLM response, tolerating markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    # Find the first [ ... ] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [{"slug": "concept-unknown", "title": "Unknown", "type": "concept"}]
    try:
        topics = json.loads(match.group())
        return [t for t in topics if isinstance(t, dict) and "slug" in t and "title" in t]
    except json.JSONDecodeError:
        return [{"slug": "concept-unknown", "title": "Unknown", "type": "concept"}]


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
