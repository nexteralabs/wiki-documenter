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

    llm_fn: async (prompt: str) -> tuple[str, int]  — (response_text, tokens_used)
    """

    def __init__(self, wiki_root: Path, llm_fn: Callable[[str], Awaitable[tuple[str, int]]]):
        self.storage = WikiStorage(wiki_root)
        self.fetcher = ArticleFetcher()
        self.llm_fn = llm_fn

    async def document(self, url: str, title: str) -> dict:
        content = await self.fetcher.fetch(url)
        return await self.document_with_content(url, title, content)

    async def document_with_content(self, url: str, title: str, content: str) -> dict:
        """
        Full document pipeline:
        1. Save raw article content
        2. Create sources/source-<slug>.md — full article summary with TL;DR + takeaways
        3. Identify 2-4 main concepts the article covers
        4. For each concept: append a "From [article]" section to its wiki page
        5. Source page links to all concept pages; concept pages link back to source
        """
        raw_slug = _slugify(title)
        source_slug = f"sources/source-{raw_slug}"
        total_tokens = 0

        self.storage.save_raw(raw_slug, content)

        # Step 1 — write the source summary page
        source_prompt = _build_source_page_prompt(title, url, content)
        source_markdown, tokens = await self.llm_fn(source_prompt)
        total_tokens += tokens
        self.storage.write_page(source_slug, source_markdown)

        source_page_entry = {
            "page": source_slug,
            "title": f"Source: {title}",
            "type": "source",
            "action": "created",
            "path": str(self.storage._wiki_path(source_slug)),
        }

        # Step 2 — identify concepts
        topics, tokens_topics = await self._identify_topics(title, content)
        total_tokens += tokens_topics

        # Step 3 — append "From [article]" section to each concept page
        concept_pages = []
        topic_wikilinks = []

        for topic in topics:
            existing = self.storage.read_page(topic["slug"])
            filename = topic["slug"].split("/")[-1]
            topic_wikilinks.append(f"[[{filename}\\|{topic['title']}]]")

            prompt = _build_concept_section_prompt(
                topic_title=topic["title"],
                topic_slug=topic["slug"],
                article_title=title,
                article_url=url,
                source_slug=source_slug,
                article_content=content,
                existing=existing,
            )
            section_markdown, tokens = await self.llm_fn(prompt)
            total_tokens += tokens

            if existing:
                # Append new section to existing page
                updated = existing.rstrip() + "\n\n" + section_markdown.strip()
                self.storage.write_page(topic["slug"], updated)
                action = "enhanced"
            else:
                # New page — write a stub with the section
                stub = _build_concept_stub(topic["title"], topic["slug"], topic["type"]) + "\n\n" + section_markdown.strip()
                self.storage.write_page(topic["slug"], stub)
                action = "created"

            concept_pages.append({
                "page": topic["slug"],
                "title": topic["title"],
                "type": topic["type"],
                "action": action,
                "path": str(self.storage._wiki_path(topic["slug"])),
            })

        # Step 4 — patch source page to include concept wikilinks
        if topic_wikilinks:
            source_content = self.storage.read_page(source_slug) or source_markdown
            topics_line = "\n\n## Concepts Covered\n" + "\n".join(f"- {wl}" for wl in topic_wikilinks)
            self.storage.write_page(source_slug, source_content.rstrip() + topics_line)

        all_pages = [source_page_entry] + concept_pages
        self.storage.append_new_content(title, url, all_pages)

        return {
            "pages": all_pages,
            "source_page": source_slug,
            "raw": raw_slug,
            "source_title": title,
            "source_url": url,
            "tokens_used": total_tokens,
        }

    async def research(self, topic: str) -> dict:
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
        existing_pages = self.storage.list_pages()
        prompt = _build_topics_prompt(title, content, existing_pages)
        raw, tokens = await self.llm_fn(prompt)
        return _parse_topics_json(raw), tokens


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_NO_FENCE = "CRITICAL: Output raw markdown ONLY. No code fences (``` or ~~~) anywhere in the output.\n\n"

_FRONTMATTER = """\
---
title: "{title}"
type: {type}
tags: [{tags}]
sources: []
updated: "{today}"
---
"""

_FORMAT_RULES = """\
FORMATTING RULES:
- Use markdown tables for comparisons or option lists
- Use `---` to separate major sections
- Use code blocks with language tags for code/CLI
- Use `- ` bullet lists (not `* `), short and scannable
- `###` subheadings for distinct sub-topics
- Cross-link related pages with [[wikilink]] notation
- No long prose paragraphs — break into scannable chunks
- Include version numbers, flag names, real numbers where relevant
"""


def _build_source_page_prompt(title: str, url: str, content: str) -> str:
    return f"""\
{_NO_FENCE}Write a source summary wiki page for this article.

Output this exact structure (raw markdown, no code fences):

---
title: "{title}"
type: source
tags: [source, article]
sources: []
updated: "{_TODAY}"
---

## TL;DR
[2-3 sentences. What does this article argue overall? What's the main point?]

## Key Takeaways
- [Most important actionable insight]
- [Second insight]
- [Third insight]
- [Add up to 6-8 bullets total — concrete, specific, not generic]

## Summary
[3-5 short paragraphs or a structured breakdown covering the article's main sections. \
Preserve the author's reasoning and narrative. Use ### subheadings for major sections. \
Include specific numbers, tool names, and examples mentioned.]

## Sources
- [{title}]({url}) — retrieved {_TODAY}

---

Article to summarize:
Title: {title}
URL: {url}

Content:
{content[:10000]}
"""


def _build_concept_section_prompt(
    topic_title: str,
    topic_slug: str,
    article_title: str,
    article_url: str,
    source_slug: str,
    article_content: str,
    existing: str | None,
) -> str:
    source_filename = source_slug.split("/")[-1]
    existing_note = ""
    if existing:
        existing_note = f"\nExisting page already has content — do NOT repeat what is already there.\n"

    return f"""\
{_NO_FENCE}Write a section to add to the wiki page for "{topic_title}".

This section documents what the article "{article_title}" specifically says about this concept.
{existing_note}
Output ONLY the section markdown (no frontmatter, no full page). Start with the heading:

## From [[{source_filename}\\|{article_title}]] ({_TODAY})

Then 3-8 bullet points or short structured content covering:
- What this article specifically says about {topic_title}
- Any concrete numbers, examples, or techniques mentioned
- Key trade-offs or recommendations from this article

{_FORMAT_RULES}

Article content (extract only what's relevant to {topic_title}):
{article_content[:8000]}
"""


def _build_concept_stub(title: str, slug: str, page_type: str) -> str:
    tags = slug.split("/")[0] if "/" in slug else "general"
    return f"""\
---
title: "{title}"
type: {page_type}
tags: [{tags}]
sources: []
updated: "{_TODAY}"
---

## TL;DR
_Summary to be written as more sources are added._

## Details
_See article sections below for current knowledge on this topic._
"""


def _build_topics_prompt(title: str, content: str, existing_pages: list[str]) -> str:
    pages_list = "\n".join(f"- {p}" for p in existing_pages[:150]) if existing_pages else "(none yet)"
    return f"""\
You are identifying the main concepts an article covers, to update the wiki.

Article title: {title}
Article content (first 5000 chars):
{content[:5000]}

---
Existing wiki pages (slugs):
{pages_list}

---
Identify 2-4 concepts that are the MAIN subjects of this article — topics it actually \
teaches or gives real guidance on. Not passing mentions.

If an existing page matches (same or very similar slug), USE that slug to enhance it.

Return ONLY a JSON array:
[
  {{"slug": "rag/technique-prompt-caching", "title": "Prompt Caching", "type": "technique"}},
  {{"slug": "agents/concept-agent-memory", "title": "Agent Memory", "type": "concept"}}
]

Rules:
- slug: folder/type-name (ALWAYS include folder)
- Folders: rag | agents | observability | security | containers | python | spark | llm | devops
- type prefix: tool- | technique- | concept- | compare- | pattern- | implementation-
- type field: tool | technique | concept | comparison | overview
- 2-4 topics MAX
- Prefer existing slugs over near-duplicates
"""


def _build_research_prompt(topic: str, existing: str | None) -> str:
    existing_section = f"\n## Existing page — preserve all, add/update:\n\n{existing}\n\n---\n" if existing else ""
    return f"""\
{_NO_FENCE}Research "{topic}" and write a comprehensive wiki page.

{_FRONTMATTER.format(title=topic, type="concept", tags=_slugify(topic), today=_TODAY)}

## TL;DR
[2-3 sentences — the essential takeaway]

## Details
[Comprehensive current content — use ### subheadings, tables, code blocks]

## Sources
- [Source title](url) — retrieved {_TODAY}

{_FORMAT_RULES}
{existing_section}
Use your search/grounding capabilities for current info (2025-2026).
Topic: {topic}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_topics_json(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        topics = json.loads(match.group())
        return [t for t in topics if isinstance(t, dict) and "slug" in t and "title" in t]
    except json.JSONDecodeError:
        return []


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
