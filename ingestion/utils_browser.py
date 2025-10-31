import os
from typing import Optional


def fetch_html_playwright(url: str, timeout_ms: int = 30000) -> Optional[str]:
    """Fetch page HTML using Playwright (sync). Returns None if playwright not installed.

    Requires `pip install playwright` and one-time `playwright install chromium`.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=os.getenv(
                "BROWSER_USER_AGENT",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            )
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(url, wait_until="load")
        html = page.content()
        context.close()
        browser.close()
        return html


async def fetch_html_playwright_async(url: str, timeout_ms: int = 30000) -> Optional[str]:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=os.getenv(
                "BROWSER_USER_AGENT",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            )
        )
        page = await context.new_page()
        page.set_default_timeout(timeout_ms)
        await page.goto(url, wait_until="load")
        html = await page.content()
        await context.close()
        await browser.close()
        return html


