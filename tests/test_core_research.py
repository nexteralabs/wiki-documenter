import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from wiki_documenter.core import WikiDocumenter


@pytest.fixture
def wiki_root(tmp_path):
    return tmp_path


@pytest.fixture
def llm_fn():
    return AsyncMock(return_value="""---
title: "GraphQL"
type: concept
tags: [graphql, api, query-language]
sources: []
updated: "2026-06-01"
---

# GraphQL

GraphQL is a query language for APIs developed by Meta.

## Sources
- https://graphql.org — retrieved 2026-06-01
""")


@pytest.fixture
def documenter(wiki_root, llm_fn):
    return WikiDocumenter(wiki_root=wiki_root, llm_fn=llm_fn)


@pytest.mark.asyncio
async def test_research_creates_page(documenter, wiki_root):
    result = await documenter.research("GraphQL")

    assert result["action"] == "created"
    assert result["page"] == "graphql"
    assert (wiki_root / "wiki" / "graphql.md").exists()


@pytest.mark.asyncio
async def test_research_enhances_existing_page(documenter, wiki_root):
    wiki_dir = wiki_root / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "graphql.md").write_text("# Existing GraphQL content\nOld info.")

    result = await documenter.research("GraphQL")

    assert result["action"] == "enhanced"
    call_args = documenter.llm_fn.call_args[0][0]
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
async def test_research_returns_page_and_path(documenter, wiki_root):
    result = await documenter.research("GraphQL")

    assert "page" in result
    assert "action" in result
    assert "path" in result
    assert result["path"].endswith("graphql.md")
