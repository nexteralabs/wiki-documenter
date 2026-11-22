import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx


class FetchError(Exception):
    pass


# Sites like VentureBeat block a bot User-Agent (403/429) but serve a browser.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
BLOCKED_STATUSES = (401, 403, 429)


class ArticleFetcher:
    PROFILE_DIR = Path.home() / ".playwright-profiles"

    async def fetch(self, url: str) -> str:
        # 1. Try httpx
        status = None
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers=BROWSER_HEADERS)
                status = response.status_code
                if status not in BLOCKED_STATUSES:
                    response.raise_for_status()
                    return self._extract_text(response.text)
        except httpx.HTTPStatusError:
            pass

        # 2. Playwright fallback
        domain = self._domain(url)
        profile_path = self.PROFILE_DIR / domain
        if not profile_path.exists():
            raise FetchError(
                f"HTTP {status} for {url}. No playwright profile found at {profile_path}. "
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
