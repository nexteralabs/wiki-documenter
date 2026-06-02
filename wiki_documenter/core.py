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
    """
    Document tech articles and research topics into an Obsidian wiki vault.

    llm_fn must be an async callable: (prompt: str) -> tuple[str, int]
    returning (response_text, tokens_used).  Pass tokens_used=0 if unknown.
    """

    def __init__(self, wiki_root: Path, llm_fn: Callable[[str], Awaitable[tuple[str, int]]]):
        self.storage = WikiStorage(wiki_root)
        self.fetcher = ArticleFetcher()
        self.llm_fn = llm_fn

    async def document(self, url: str, title: str) -> dict:
        """Fetch URL, identify topics, write/enhance one wiki page per topic."""
        content = await self.fetcher.fetch(url)
        return await self.document_with_content(url, title, content)

    async def document_with_content(self, url: str, title: str, content: str) -> dict:
        """Same as document() but with pre-fetched content — skips the HTTP fetch."""
        raw_slug = _slugify(title)
        self.storage.save_raw(raw_slug, content)

        topics, tokens_topics = await self._identify_topics(title, content)
        total_tokens = tokens_topics

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
            wiki_markdown, tokens = await self.llm_fn(prompt)
            total_tokens += tokens
            path = self.storage.write_page(topic["slug"], wiki_markdown)
            action = "enhanced" if existing else "created"
            pages.append({
                "page": topic["slug"],
                "title": topic["title"],
                "type": topic["type"],
                "action": action,
                "path": str(path),
            })

        self.storage.append_new_content(title, url, pages)
        return {
            "pages": pages,
            "raw": raw_slug,
            "source_title": title,
            "source_url": url,
            "tokens_used": total_tokens,
        }

    async def research(self, topic: str) -> dict:
        """Research a topic using LLM search grounding and write/enhance a wiki page."""
        slug = _slugify(topic)
        existing = self.storage.read_page(slug)
        prompt = _build_research_prompt(topic, existing)
        wiki_markdown, tokens = await self.llm_fn(prompt)
        path = self.storage.write_page(slug, wiki_markdown)
        action = "enhanced" if existing else "created"

        pages = [{"page": slug, "title": topic, "type": "concept", "action": action, "path": str(path)}]
        self.storage.append_new_content(f"research: {topic}", "", pages)

        return {"pages": pages, "source_title": topic, "source_url": "", "tokens_used": tokens}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _identify_topics(self, title: str, content: str) -> tuple[list[dict], int]:
        """Ask the LLM to extract distinct topics. Returns (topics, tokens_used)."""
        existing_pages = self.storage.list_pages()
        prompt = _build_topics_prompt(title, content, existing_pages)
        raw, tokens = await self.llm_fn(prompt)
        return _parse_topics_json(raw), tokens


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_PAGE_FORMAT = """\
CRITICAL: Output raw markdown ONLY. Do NOT wrap anything in code fences (``` or ~~~). \
Do NOT add ```yaml or ```markdown anywhere. The output is written directly to a .md file.

Every wiki page MUST start with YAML frontmatter (no surrounding code fences), then two sections:

---
title: "Human Readable Title"
type: tool | technique | concept | comparison | overview
tags: [relevant, tags]
sources: []
updated: "{today}"
---

## TL;DR
2–4 sentences max. The single most useful takeaway — what a busy engineer needs in 30 seconds.

## Details

[Body here — see formatting rules below]

## Sources
- [Title](https://url) — retrieved {today}

---

FORMATTING RULES (follow these exactly):
- Use markdown tables for comparisons, options, or config parameters
- Use `---` horizontal rules to separate major sections
- Use code blocks with language tags for any code, config, or CLI commands
- Use `- ` bullet lists (not `* `), keep them short and scannable
- Subheadings with `###` for distinct sub-topics
- Cross-link related wiki pages with [[wikilink]] notation
- NO long prose paragraphs — break everything into scannable chunks
- Be concrete: include version numbers, actual flag names, real numbers
"""


def _build_topics_prompt(title: str, content: str, existing_pages: list[str]) -> str:
    pages_list = "\n".join(f"- {p}" for p in existing_pages[:150]) if existing_pages else "(none yet)"
    return f"""\
You are deciding which wiki pages to create or update based on a tech article.

Article title: {title}

Article content (first 5000 chars):
{content[:5000]}

---

## Existing wiki pages (slugs):
{pages_list}

---

## Your job

Identify the 2–4 topics that are the MAIN subjects of this article — topics the article \
actually teaches, explains in depth, or gives actionable guidance on.

DO NOT include:
- Topics merely mentioned in passing or as context
- Generic/broad concepts (e.g. "AI", "machine learning", "software engineering")
- Topics already well-covered by an existing page UNLESS this article adds significant new content

If an existing page matches a topic (same or very similar slug), USE that slug so we enhance \
it instead of creating a duplicate.

Return ONLY a JSON array, no other text:
[
  {{"slug": "rag/technique-prompt-caching", "title": "Prompt Caching", "type": "technique"}},
  {{"slug": "rag/technique-semantic-caching", "title": "Semantic Caching", "type": "technique"}}
]

Rules:
- slug: folder/type-name — ALWAYS include a folder prefix
- Folders: rag | agents | observability | security | containers | python | spark | d365 | confluence | controlm
  Create a new folder name if none fit (e.g. "llm", "data", "devops")
- type prefix: tool- | technique- | concept- | compare- | pattern- | implementation-
- type field: tool | technique | concept | comparison | overview
- 2–4 topics MAX — quality over quantity
- Prefer reusing an existing slug over creating a near-duplicate
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
