import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from wiki_documenter.fetcher import ArticleFetcher, FetchError

@pytest.fixture
def fetcher():
    return ArticleFetcher()

@pytest.mark.asyncio
async def test_fetch_normal_url_returns_text(fetcher):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><p>Article content here</p></body></html>"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await fetcher.fetch("https://example.com/article")
        assert "Article content here" in result

@pytest.mark.asyncio
async def test_playwright_fallback_used_when_profile_exists(fetcher, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"

    # Create a fake profile dir for the domain
    profile_dir = tmp_path / "example.com"
    profile_dir.mkdir()

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch.object(fetcher, "PROFILE_DIR", tmp_path), \
         patch("asyncio.create_subprocess_exec") as mock_proc:

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"Page content from playwright", b""))
        mock_process.returncode = 0
        mock_proc.return_value = mock_process

        result = await fetcher.fetch("https://example.com/article")
        # playwright-cli was invoked (at least once — first call is the fetch, second is close)
        assert mock_proc.call_count >= 1
        first_call_args = mock_proc.call_args_list[0][0]
        assert "playwright-cli" in first_call_args
        assert str(profile_dir) in str(mock_proc.call_args_list[0])

@pytest.mark.asyncio
async def test_no_fallback_without_profile_raises_fetch_error(fetcher, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch.object(fetcher, "PROFILE_DIR", tmp_path):  # tmp_path has no example.com subdir

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("https://example.com/article")
        assert "playwright" in str(exc_info.value).lower() or "profile" in str(exc_info.value).lower()
