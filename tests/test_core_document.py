import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from wiki_documenter.core import WikiDocumenter

_TOPICS_JSON = '[{"slug": "rag/tool-graphql", "title": "GraphQL", "type": "tool"}]'

_SOURCE_PAGE = """---
title: "GraphQL Guide"
type: source
tags: [source, article]
sources: []
updated: "2026-06-01"
---

## TL;DR
GraphQL is a query language for APIs that lets clients request exactly the data they need.

## Key Takeaways
- GraphQL reduces over-fetching
- Strongly typed schema

## Summary
The article covers GraphQL fundamentals.

## Sources
- [GraphQL Guide](https://example.com/graphql) — retrieved 2026-06-01
"""

_CONCEPT_SECTION = """## From [[source-graphql-guide\\|GraphQL Guide]] (2026-06-01)
- GraphQL lets clients specify exactly what data they need
- Uses a strongly typed schema
"""


@pytest.fixture
def wiki_root(tmp_path):
    return tmp_path


@pytest.fixture
def llm_fn():
    # Calls: source page, topics, concept section
    return AsyncMock(side_effect=[
        (_SOURCE_PAGE, 500),   # source page
        (_TOPICS_JSON, 100),   # topic extraction
        (_CONCEPT_SECTION, 300),  # concept section
    ])


@pytest.fixture
def documenter(wiki_root, llm_fn):
    d = WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)
    d.fetcher.fetch = AsyncMock(return_value="Full article content about GraphQL")
    return d


@pytest.mark.asyncio
async def test_document_creates_source_page(documenter, wiki_root):
    result = await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    source = wiki_root / "wiki" / "sources" / "source-graphql-guide.md"
    assert source.exists()


@pytest.mark.asyncio
async def test_document_source_page_is_first_in_pages(documenter):
    result = await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    assert result["pages"][0]["type"] == "source"
    assert "source-graphql-guide" in result["pages"][0]["page"]


@pytest.mark.asyncio
async def test_document_creates_concept_section(documenter, wiki_root):
    await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    concept = wiki_root / "wiki" / "rag" / "tool-graphql.md"
    assert concept.exists()
    content = concept.read_text()
    assert "From" in content
    assert "GraphQL Guide" in content


@pytest.mark.asyncio
async def test_document_enhances_existing_concept(wiki_root):
    (wiki_root / "wiki" / "rag").mkdir(parents=True, exist_ok=True)
    existing = wiki_root / "wiki" / "rag" / "tool-graphql.md"
    existing.write_text("---\ntitle: GraphQL\n---\n\n## TL;DR\nExisting content.")

    llm_fn = AsyncMock(side_effect=[
        (_SOURCE_PAGE, 500),
        (_TOPICS_JSON, 100),
        (_CONCEPT_SECTION, 300),
    ])
    d = WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)
    d.fetcher.fetch = AsyncMock(return_value="content")

    result = await d.document_with_content("https://example.com/graphql", "GraphQL Guide", "content")

    concept_pages = [p for p in result["pages"] if p["type"] != "source"]
    assert concept_pages[0]["action"] == "enhanced"
    content = existing.read_text()
    assert "Existing content" in content
    assert "From" in content


@pytest.mark.asyncio
async def test_document_source_page_links_concepts(documenter, wiki_root):
    await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    source = (wiki_root / "wiki" / "sources" / "source-graphql-guide.md").read_text()
    assert "Concepts Covered" in source
    assert "tool-graphql" in source


@pytest.mark.asyncio
async def test_document_saves_raw(documenter, wiki_root):
    await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    assert (wiki_root / "raw" / "graphql-guide.md").exists()


@pytest.mark.asyncio
async def test_document_returns_token_count(documenter):
    result = await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    assert result["tokens_used"] == 900  # 500 + 100 + 300


@pytest.mark.asyncio
async def test_document_updates_new_content_log(documenter, wiki_root):
    await documenter.document_with_content(
        "https://example.com/graphql", "GraphQL Guide", "article content")
    log = wiki_root / "new-content.md"
    assert log.exists()
    assert "GraphQL Guide" in log.read_text()
