import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx


class FetchError(Exception):
    pass


class ArticleFetcher:
    PROFILE_DIR = Path.home() / ".playwright-profiles"

    async def fetch(self, url: str) -> str:
        # 1. Try httpx
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; WikiDocumenter/1.0)"}
                response = await client.get(url, headers=headers)
                if response.status_code not in (401, 403):
                    response.raise_for_status()
                    return self._extract_text(response.text)
        except httpx.HTTPStatusError:
            pass

        # 2. Playwright fallback
        domain = self._domain(url)
        profile_path = self.PROFILE_DIR / domain
        if not profile_path.exists():
            raise FetchError(
                f"HTTP 403 for {url}. No playwright profile found at {profile_path}. "
                f"Login first: playwright-cli open {url} --persistent --profile={profile_path}"
            )

        return await self._fetch_with_playwright(url, profile_path)

    def _domain(self, url: str) -> str:
        netloc = urlparse(url).netloc
        return netloc.removeprefix("www.")

    def _extract_text(self, html: str) -> str:
        # Remove script/style tags and extract visible text
        html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<[^>]+>', ' ', html)
        html = re.sub(r'\s+', ' ', html)
        return html.strip()

    async def _fetch_with_playwright(self, url: str, profile_path: Path) -> str:
        proc = await asyncio.create_subprocess_exec(
            "playwright-cli", "--raw", "open", url,
            "--persistent", f"--profile={profile_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise FetchError(f"playwright-cli failed: {stderr.decode()}")

        # Close the browser
        await asyncio.create_subprocess_exec("playwright-cli", "close")
        return stdout.decode()
