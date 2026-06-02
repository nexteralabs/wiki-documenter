import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from wiki_documenter.core import WikiDocumenter

_TOPICS_JSON = '[{"slug": "tool-graphql", "title": "GraphQL", "type": "tool"}]'

_PAGE_CONTENT = """---
title: "GraphQL"
type: tool
tags: [graphql, api]
sources: []
updated: "2026-06-01"
---

## TL;DR
GraphQL is a query language for APIs.

## Details
GraphQL allows clients to request exactly the data they need.

## Sources
- https://example.com/graphql — retrieved 2026-06-01
"""


@pytest.fixture
def wiki_root(tmp_path):
    return tmp_path


@pytest.fixture
def llm_fn():
    # First call returns topics JSON, subsequent calls return page content
    mock = AsyncMock(side_effect=[_TOPICS_JSON, _PAGE_CONTENT])
    return mock


@pytest.fixture
def documenter(wiki_root, llm_fn):
    d = WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)
    d.fetcher.fetch = AsyncMock(return_value="Full article content about GraphQL")
    return d


@pytest.mark.asyncio
async def test_document_returns_pages_list(documenter):
    result = await documenter.document("https://example.com/graphql", "GraphQL Guide")

    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["pages"][0]["action"] == "created"
    assert result["pages"][0]["page"] == "tool-graphql"


@pytest.mark.asyncio
async def test_document_creates_wiki_page(documenter, wiki_root):
    await documenter.document("https://example.com/graphql", "GraphQL Guide")

    wiki_page = wiki_root / "wiki" / "tool-graphql.md"
    assert wiki_page.exists()


@pytest.mark.asyncio
async def test_document_saves_raw(documenter, wiki_root):
    await documenter.document("https://example.com/graphql", "GraphQL Guide")

    raw_file = wiki_root / "raw" / "graphql-guide.md"
    assert raw_file.exists()


@pytest.mark.asyncio
async def test_document_enhances_existing_page(wiki_root):
    # Pre-create existing wiki page
    wiki_dir = wiki_root / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "tool-graphql.md").write_text("# Existing GraphQL content")

    llm_fn = AsyncMock(side_effect=[_TOPICS_JSON, _PAGE_CONTENT])
    d = WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)
    d.fetcher.fetch = AsyncMock(return_value="New article content")

    result = await d.document("https://example.com/graphql", "GraphQL Guide")

    assert result["pages"][0]["action"] == "enhanced"
    # LLM was called with existing content in the prompt (second call)
    second_call_prompt = llm_fn.call_args_list[1][0][0]
    assert "Existing GraphQL content" in second_call_prompt


@pytest.mark.asyncio
async def test_document_updates_new_content_log(documenter, wiki_root):
    await documenter.document("https://example.com/graphql", "GraphQL Guide")

    log_file = wiki_root / "new-content.md"
    assert log_file.exists()
    content = log_file.read_text()
    assert "GraphQL Guide" in content
    assert "tool-graphql" in content


@pytest.mark.asyncio
async def test_document_returns_source_info(documenter):
    result = await documenter.document("https://example.com/graphql", "GraphQL Guide")

    assert result["source_title"] == "GraphQL Guide"
    assert result["source_url"] == "https://example.com/graphql"
    assert result["raw"] == "graphql-guide"
