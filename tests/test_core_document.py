import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from wiki_documenter.core import WikiDocumenter


@pytest.fixture
def wiki_root(tmp_path):
    return tmp_path


@pytest.fixture
def llm_fn():
    return AsyncMock(return_value="""---
title: "GraphQL Guide"
type: tool
tags: [graphql, api]
sources: []
updated: "2026-06-01"
---

# GraphQL Guide

GraphQL is a query language for APIs.

## Sources
- https://example.com/graphql — retrieved 2026-06-01
""")


@pytest.fixture
def documenter(wiki_root, llm_fn):
    d = WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)
    # Patch the fetcher so we don't make real HTTP calls
    d.fetcher.fetch = AsyncMock(return_value="Full article content about GraphQL")
    return d


@pytest.mark.asyncio
async def test_document_creates_new_page(documenter, wiki_root):
    result = await documenter.document("https://example.com/graphql", "GraphQL Guide")

    assert result["action"] == "created"
    assert result["page"] == "graphql-guide"
    wiki_page = wiki_root / "wiki" / "graphql-guide.md"
    assert wiki_page.exists()


@pytest.mark.asyncio
async def test_document_enhances_existing_page(documenter, wiki_root):
    # Pre-create existing wiki page
    wiki_dir = wiki_root / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "graphql-guide.md").write_text("# Existing GraphQL content")

    result = await documenter.document("https://example.com/graphql", "GraphQL Guide")

    assert result["action"] == "enhanced"
    # LLM was called with existing content in the prompt
    call_args = documenter.llm_fn.call_args[0][0]
    assert "Existing GraphQL content" in call_args


@pytest.mark.asyncio
async def test_document_saves_raw(documenter, wiki_root):
    await documenter.document("https://example.com/graphql", "GraphQL Guide")

    raw_file = wiki_root / "raw" / "graphql-guide.md"
    assert raw_file.exists()
    assert "Full article content about GraphQL" in raw_file.read_text()


@pytest.mark.asyncio
async def test_document_passes_article_to_llm(documenter):
    await documenter.document("https://example.com/graphql", "GraphQL Guide")

    call_args = documenter.llm_fn.call_args[0][0]
    assert "GraphQL Guide" in call_args
    assert "Full article content about GraphQL" in call_args


@pytest.mark.asyncio
async def test_document_returns_path(documenter, wiki_root):
    result = await documenter.document("https://example.com/graphql", "GraphQL Guide")

    assert "path" in result
    assert result["path"].endswith("graphql-guide.md")
