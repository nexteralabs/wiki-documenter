import pytest
from pathlib import Path
from wiki_documenter.storage import WikiStorage


@pytest.fixture
def storage(tmp_path):
    return WikiStorage(wiki_root=tmp_path)


def test_write_read_page(storage):
    storage.write_page("tool-graphql", "# GraphQL\ncontent here")
    result = storage.read_page("tool-graphql")
    assert result == "# GraphQL\ncontent here"


def test_read_missing_page_returns_none(storage):
    assert storage.read_page("does-not-exist") is None


def test_save_raw(storage, tmp_path):
    storage.save_raw("graphql-article", "raw article content")
    raw_file = tmp_path / "raw" / "graphql-article.md"
    assert raw_file.exists()
    assert raw_file.read_text() == "raw article content"


def test_search_pages(storage):
    storage.write_page("tool-graphql", "GraphQL is a query language for APIs")
    storage.write_page("tool-rest", "REST is an architectural style")
    results = storage.search("graphql", max_results=5)
    assert len(results) > 0
    assert results[0]["name"] == "tool-graphql"
    assert "score" in results[0]
    assert "snippet" in results[0]


def test_search_returns_empty_for_no_match(storage):
    storage.write_page("tool-graphql", "GraphQL content")
    results = storage.search("kubernetes", max_results=5)
    assert results == []


def test_list_pages(storage):
    storage.write_page("tool-graphql", "content")
    storage.write_page("concept-api", "content")
    pages = storage.list_pages()
    assert "tool-graphql" in pages
    assert "concept-api" in pages
    assert pages == sorted(pages)


def test_path_traversal_blocked(storage):
    with pytest.raises(ValueError):
        storage.write_page("../evil", "content")
