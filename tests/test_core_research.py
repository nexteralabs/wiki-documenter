import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from wiki_documenter.core import WikiDocumenter

_PAGE_CONTENT = """---
title: "GraphQL"
type: concept
tags: [graphql, api, query-language]
sources: []
updated: "2026-06-01"
---

## TL;DR
GraphQL is a query language for APIs developed by Meta.

## Details
GraphQL allows clients to request exactly what they need.

## Sources
- https://graphql.org — retrieved 2026-06-01
"""


@pytest.fixture
def wiki_root(tmp_path):
    return tmp_path


@pytest.fixture
def llm_fn():
    return AsyncMock(return_value=_PAGE_CONTENT)


@pytest.fixture
def documenter(wiki_root, llm_fn):
    return WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)


@pytest.mark.asyncio
async def test_research_returns_pages_list(documenter):
    result = await documenter.research("GraphQL")

    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["pages"][0]["action"] == "created"


@pytest.mark.asyncio
async def test_research_creates_page(documenter, wiki_root):
    await documenter.research("GraphQL")

    assert (wiki_root / "wiki" / "graphql.md").exists()


@pytest.mark.asyncio
async def test_research_enhances_existing_page(wiki_root):
    wiki_dir = wiki_root / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "graphql.md").write_text("# Existing GraphQL content\nOld info.")

    llm_fn = AsyncMock(return_value=_PAGE_CONTENT)
    d = WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)
    result = await d.research("GraphQL")

    assert result["pages"][0]["action"] == "enhanced"
    call_args = llm_fn.call_args[0][0]
    assert "Existing GraphQL content" in call_args


@pytest.mark.asyncio
async def test_research_prompt_contains_topic(documenter):
    await documenter.research("Kubernetes Operators")

    call_args = documenter.llm_fn.call_args[0][0]
    assert "Kubernetes Operators" in call_args


@pytest.mark.asyncio
async def test_research_prompt_instructs_web_search(documenter):
    await documenter.research("GraphQL")

    call_args = documenter.llm_fn.call_args[0][0]
    assert "search" in call_args.lower() or "grounding" in call_args.lower()


@pytest.mark.asyncio
async def test_research_updates_new_content_log(documenter, wiki_root):
    await documenter.research("GraphQL")

    log_file = wiki_root / "new-content.md"
    assert log_file.exists()
    assert "graphql" in log_file.read_text()
