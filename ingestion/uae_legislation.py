import os
import re
import hashlib
import logging
import asyncio
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from .utils_browser import fetch_html_playwright, fetch_html_playwright_async

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None
from pypdf import PdfReader
try:
    import boto3
except Exception:
    boto3 = None


LOGGER_NAME = "uae_legislation_ingest"
LOG_FILE = os.getenv("UAE_LEGISLATION_LOG_FILE", "uae_legislation_ingest.log")


logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)


DEFAULT_SECTOR_URL = "https://uaelegislation.gov.ae/ar/legislations?sector=47"


@dataclass
class PDFMeta:
    url: str
    title: Optional[str]
    year: Optional[str]
    law_number: Optional[str]
    source_page: str


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch_html(url: str, timeout: int = 25, use_browser: bool = False) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AkkioBot/1.0; +https://example.com/bot)",
        "Accept-Language": "ar,en;q=0.9",
    }
    try:
        if use_browser or os.getenv("ENABLE_PLAYWRIGHT", "0").lower() in {"1", "true", "yes"}:
            html = fetch_html_playwright(url)
            if html:
                return html
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"HTTP/Browser fetch failed for {url}: {e}")
        return None


def discover_pdfs(list_url: str = DEFAULT_SECTOR_URL, timeout: int = 25, use_browser: bool = False, max_detail_pages: int = 50) -> List[PDFMeta]:
    """Discover PDF links from the sector listing and detail pages.

    Strategy: parse the listing page for anchors linking to legislation details, then fetch those pages
    and collect any .pdf links. Also directly collect .pdf links present on the listing page.
    """
    start = time.perf_counter()
    metas: List[PDFMeta] = []
    try:
        logger.info(f"Fetching listing page: {list_url}")
        html = _fetch_html(list_url, timeout=timeout, use_browser=use_browser)
        if not html:
            logger.error("Failed to fetch listing HTML")
            return []
        soup = BeautifulSoup(html, "lxml")

        # Collect all anchors from listing page
        anchors = soup.find_all("a", href=True)
        detail_links = []
        base_host = urlparse(list_url).netloc
        for a in anchors:
            href = a["href"]
            abs_url = _absolutize(list_url, href)
            lower_abs = abs_url.lower()
            if lower_abs.endswith(".pdf") or "download" in lower_abs:
                metas.append(PDFMeta(
                    url=abs_url,
                    title=_clean_text(a.get_text()),
                    year=_extract_year(a.get_text()),
                    law_number=_extract_law_number(a.get_text()),
                    source_page=list_url,
                ))
            else:
                # follow same-domain legislation item pages
                p = urlparse(abs_url)
                path_lower = (p.path or "").lower()
                if p.netloc == base_host and ("/legislation/" in path_lower or "/legislations/" in path_lower):
                    detail_links.append(abs_url)

        # Limit detail pages
        detail_links = list(dict.fromkeys(detail_links))[:max_detail_pages]
        logger.info(f"Discovered {len(detail_links)} detail pages from listing (limited)")

        # Visit detail pages and collect .pdf
        for i, durl in enumerate(detail_links):
            try:
                dhtml = _fetch_html(durl, timeout=timeout, use_browser=use_browser)
                if not dhtml:
                    continue
                dsoup = BeautifulSoup(dhtml, "lxml")
                for a in dsoup.find_all("a", href=True):
                    href = a["href"]
                    abs2 = _absolutize(durl, href)
                    lower_abs2 = abs2.lower()
                    if lower_abs2.endswith(".pdf") or "download" in lower_abs2 or "file" in lower_abs2:
                        metas.append(PDFMeta(
                            url=abs2,
                            title=_clean_text(a.get_text()) or _find_title(dsoup),
                            year=_extract_year(dsoup.get_text()),
                            law_number=_extract_law_number(dsoup.get_text()),
                            source_page=durl,
                        ))
            except Exception as e:
                logger.warning(f"Failed fetching detail page {durl}: {e}")

        # Remove duplicates by URL
        unique: Dict[str, PDFMeta] = {}
        for m in metas:
            unique[m.url] = m

        final = list(unique.values())
        took = time.perf_counter() - start
        logger.info(f"PDF discovery completed: {len(final)} links in {took:.2f}s")
        return final
    except Exception as e:
        logger.error(f"PDF discovery error: {e}")
        return []


async def discover_pdfs_async(list_url: str = DEFAULT_SECTOR_URL, timeout: int = 25, max_detail_pages: int = 50) -> List[PDFMeta]:
    start = time.perf_counter()
    metas: List[PDFMeta] = []
    try:
        logger.info(f"[async] Fetching listing page: {list_url}")
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=os.getenv(
                    "BROWSER_USER_AGENT",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                )
            )
            page = await context.new_page()
            page.set_default_timeout(timeout * 1000)

            # Sniff network responses that are PDFs
            sniffed_pdf_urls: Dict[str, str] = {}
            def _on_response(resp):
                try:
                    url = resp.url or ""
                    if not url:
                        return
                    headers = {k.lower(): v for k, v in (resp.headers or {}).items()}
                    ct = (headers.get("content-type") or "").lower()
                    cd = (headers.get("content-disposition") or "").lower()
                    if "pdf" in ct or ".pdf" in cd or url.lower().endswith(".pdf"):
                        sniffed_pdf_urls[url] = url
                except Exception:
                    pass
            page.on("response", _on_response)

            # Load listing page and trigger lazy content
            await page.goto(list_url, wait_until="load")
            for _ in range(6):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(400)
            html = await page.content()
            if not html:
                logger.error("[async] Failed to fetch listing HTML")
                await context.close(); await browser.close()
                return []

            soup = BeautifulSoup(html, "lxml")
            base_host = urlparse(list_url).netloc

            def collect_from_soup(soup_obj: BeautifulSoup, source: str) -> Tuple[List[str], List[PDFMeta]]:
                ds: List[str] = []
                pm: List[PDFMeta] = []
                for a in soup_obj.find_all("a", href=True):
                    href = a["href"]
                    abs_url = _absolutize(source, href)
                    lower_abs = abs_url.lower()
                    if lower_abs.endswith(".pdf") or "download" in lower_abs or "file" in lower_abs:
                        pm.append(PDFMeta(
                            url=abs_url,
                            title=_clean_text(a.get_text()),
                            year=_extract_year(a.get_text()),
                            law_number=_extract_law_number(a.get_text()),
                            source_page=source,
                        ))
                    else:
                        p = urlparse(abs_url)
                        path_lower = (p.path or "").lower()
                        if p.netloc == base_host and ("/legislation/" in path_lower or "/legislations/" in path_lower):
                            ds.append(abs_url)
                return ds, pm

            detail_links, pm_from_listing = collect_from_soup(soup, list_url)
            metas.extend(pm_from_listing)

            # Pagination pass: harvest additional detail links from a few pages
            pagination_urls = set()
            for a in soup.find_all("a", href=True):
                href = a["href"].lower()
                if "page=" in href:
                    pagination_urls.add(_absolutize(list_url, a["href"]))
            extra_pages = list(pagination_urls)[: max(0, min(5, max_detail_pages // 10))]
            for purl in extra_pages:
                try:
                    await page.goto(purl, wait_until="load")
                    for _ in range(3):
                        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(300)
                    phtml = await page.content()
                    psoup = BeautifulSoup(phtml, "lxml")
                    ds, pm2 = collect_from_soup(psoup, purl)
                    detail_links.extend(ds)
                    metas.extend(pm2)
                except Exception:
                    pass

            # Unique and limit
            detail_links = list(dict.fromkeys(detail_links))[:max_detail_pages]
            logger.info(f"[async] Discovered {len(detail_links)} detail pages from listing (limited)")

            async def is_pdf_url(u: str) -> bool:
                try:
                    r = await context.request.head(u, timeout=timeout * 1000)
                    ct = (r.headers.get("content-type") or r.headers.get("Content-Type") or "").lower()
                    cd = (r.headers.get("content-disposition") or r.headers.get("Content-Disposition") or "")
                    if "pdf" in ct or ".pdf" in cd.lower():
                        return True
                    r2 = await context.request.get(u, timeout=timeout * 1000)
                    ct2 = (r2.headers.get("content-type") or r2.headers.get("Content-Type") or "").lower()
                    cd2 = (r2.headers.get("content-disposition") or r2.headers.get("Content-Disposition") or "")
                    return ("pdf" in ct2) or (".pdf" in cd2.lower())
                except Exception:
                    return False

            async def collect_from_detail(durl: str):
                try:
                    await page.goto(durl, wait_until="load")
                    for _ in range(3):
                        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(250)
                    dhtml = await page.content()
                    dsoup = BeautifulSoup(dhtml, "lxml")
                    anchors = dsoup.find_all("a", href=True)
                    hrefs = []
                    for a in anchors:
                        hrefs.append(_absolutize(durl, a["href"]))
                        dh = a.attrs.get("data-href")
                        if dh:
                            hrefs.append(_absolutize(durl, dh))
                    # Add any sniffed pdfs during this navigation
                    for u in list(sniffed_pdf_urls.keys()):
                        hrefs.append(u)

                    candidates = [h for h in hrefs if h.lower().endswith(".pdf") or "download" in h.lower() or "file" in h.lower()]
                    if not candidates:
                        candidates = hrefs

                    sem = asyncio.Semaphore(10)
                    found_local: List[PDFMeta] = []

                    async def check(u: str):
                        async with sem:
                            if await is_pdf_url(u):
                                found_local.append(PDFMeta(
                                    url=u,
                                    title=_find_title(dsoup),
                                    year=_extract_year(dsoup.get_text()),
                                    law_number=_extract_law_number(dsoup.get_text()),
                                    source_page=durl,
                                ))

                    await asyncio.gather(*(check(u) for u in set(candidates)))
                    return found_local
                except Exception as e:
                    logger.warning(f"[async] Failed fetching detail page {durl}: {e}")
                    return []

            results = await asyncio.gather(*(collect_from_detail(u) for u in detail_links))
            for lst in results:
                metas.extend(lst)

            # Include any sniffed PDFs from listing navigations
            for u in list(sniffed_pdf_urls.keys()):
                metas.append(PDFMeta(url=u, title=None, year=None, law_number=None, source_page=list_url))

            await context.close()
            await browser.close()

        # Deduplicate
        unique: Dict[str, PDFMeta] = {}
        for m in metas:
            unique[m.url] = m
        final = list(unique.values())
        took = time.perf_counter() - start
        logger.info(f"[async] PDF discovery completed: {len(final)} links in {took:.2f}s")
        return final
    except Exception as e:
        logger.error(f"[async] PDF discovery error: {e}")
        return []


def upload_to_s3(local_path: str, bucket: Optional[str], key: Optional[str]) -> Optional[str]:
    if not bucket or not key:
        logger.warning("S3 bucket or key not provided, skipping upload")
        return None
    if boto3 is None:
        logger.warning("boto3 not installed, skipping S3 upload. Install with: pip install boto3")
        return None
    
    if not os.path.exists(local_path):
        logger.error(f"File not found for S3 upload: {local_path}")
        return None
    
    try:
        # Get credentials from environment variables only (not from credentials file)
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        
        if not aws_access_key_id or not aws_secret_access_key:
            logger.error("AWS credentials not found in environment variables. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
            return None
        
        # Create S3 client with explicit credentials from environment
        s3_config = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
        }
        if aws_region:
            s3_config["region_name"] = aws_region
        if aws_session_token:
            s3_config["aws_session_token"] = aws_session_token
        
        s3 = boto3.client("s3", **s3_config)
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        s3_url = f"s3://{bucket}/{key}"
        logger.info(f"Successfully uploaded to {s3_url}")
        return s3_url
    except Exception as e:
        error_name = type(e).__name__
        if "NoCredentialsError" in error_name or "Credentials" in error_name:
            logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        else:
            logger.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {error_name}: {e}")
        return None


def _find_title(soup: BeautifulSoup) -> Optional[str]:
    # Heuristic: find first h1/h2
    h = soup.find(["h1", "h2"]) or None
    return _clean_text(h.get_text()) if h else None


def _absolutize(base_url: str, href: str) -> str:
    try:
        return urljoin(base_url, href)
    except Exception:
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if base_url.endswith("/"):
            return base_url + href
        return base_url.rsplit("/", 1)[0] + "/" + href


def _clean_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def _extract_year(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(20\d{2}|19\d{2})", text)
    return m.group(1) if m else None


def _extract_law_number(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    # Common Arabic format contains digits; extract first number group up to 3 digits
    m = re.search(r"(?:رقم|number|no\.?)[^\d]{0,5}(\d{1,4})", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"[^\d](\d{1,4})[^\d]", " " + text + " ")
    return m2.group(1) if m2 else None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_pdf(meta: PDFMeta, download_dir: str) -> Optional[str]:
    ensure_dir(download_dir)
    try:
        fname = _safe_filename(meta.title or os.path.basename(meta.url))
        if not fname.lower().endswith(".pdf"):
            fname += ".pdf"
        local_path = os.path.join(download_dir, fname)

        # Check if already exists with same hash (before downloading)
        if os.path.exists(local_path):
            logger.info(f"File already exists, checking hash: {local_path}")
            # Continue to verify it's valid PDF by attempting download

        # Use session for cookies
        session = requests.Session()
        
        # Better headers to avoid 403 - mimic real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": meta.source_page or "https://uaelegislation.gov.ae/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Upgrade-Insecure-Requests": "1",
        }
        
        logger.info(f"Downloading PDF: {meta.url}")
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        r = session.get(meta.url, headers=headers, timeout=60, allow_redirects=True)
        
        # If 403, try with browser-based download
        if r.status_code == 403:
            logger.warning(f"403 Forbidden, trying browser-based download for {meta.url}")
            return _download_pdf_with_browser(meta, download_dir, local_path)
        
        r.raise_for_status()
        
        # Check if response is actually a PDF
        content_type = r.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not meta.url.lower().endswith(".pdf"):
            # Might be HTML redirect page
            logger.warning(f"Response is not PDF (content-type: {content_type}), trying browser download")
            return _download_pdf_with_browser(meta, download_dir, local_path)
        
        content = r.content
        # Verify it's actually a PDF (starts with %PDF)
        if not content.startswith(b"%PDF"):
            logger.warning(f"Downloaded content doesn't appear to be PDF, trying browser download")
            return _download_pdf_with_browser(meta, download_dir, local_path)
        
        new_hash = _sha256_bytes(content)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                existing_content = f.read()
                if existing_content.startswith(b"%PDF") and _sha256_bytes(existing_content) == new_hash:
                    logger.info(f"PDF unchanged, skipping: {local_path}")
                    return local_path
        
        with open(local_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved PDF: {local_path}")
        return local_path
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.warning(f"403 Forbidden for {meta.url}, trying browser-based download")
            return _download_pdf_with_browser(meta, download_dir, local_path if 'local_path' in locals() else None)
        logger.error(f"HTTP error downloading {meta.url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to download {meta.url}: {e}")
        # Last resort: try browser download
        try:
            return _download_pdf_with_browser(meta, download_dir, local_path if 'local_path' in locals() else None)
        except Exception as browser_err:
            logger.error(f"Browser download also failed for {meta.url}: {browser_err}")
            return None


def _download_pdf_with_browser(meta: PDFMeta, download_dir: str, local_path: Optional[str] = None) -> Optional[str]:
    """Fallback to browser-based PDF download when direct requests fail."""
    try:
        from playwright.sync_api import sync_playwright
        
        if local_path is None:
            fname = _safe_filename(meta.title or os.path.basename(meta.url))
            if not fname.lower().endswith(".pdf"):
                fname += ".pdf"
            local_path = os.path.join(download_dir, fname)
        
        logger.info(f"[browser] Downloading PDF via browser: {meta.url}")
        
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            
            # Set up download handling
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                accept_downloads=True,
            )
            page = context.new_page()
            page.set_default_timeout(60000)
            
            # Use request context to download with browser cookies/session
            try:
                # First, navigate to source page to establish session if needed
                if meta.source_page and meta.source_page != meta.url:
                    try:
                        page.goto(meta.source_page, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(1000)
                    except Exception:
                        pass
                
                # Now try to get the PDF via request API (uses browser context)
                response = page.request.get(meta.url)
                if response.ok:
                    content_bytes = response.body()
                    if content_bytes and content_bytes.startswith(b"%PDF"):
                        with open(local_path, "wb") as f:
                            f.write(content_bytes)
                        logger.info(f"[browser] Saved PDF via request: {local_path}")
                        context.close()
                        browser.close()
                        return local_path
                    else:
                        logger.warning(f"[browser] Response is not PDF (first bytes: {content_bytes[:50]})")
                
                # If request.get didn't work, try navigating
                logger.info(f"[browser] Trying navigation-based download")
                response = page.goto(meta.url, wait_until="load", timeout=60000)
                
                if response and response.status == 403:
                    logger.error(f"[browser] Still getting 403 for {meta.url}")
                    context.close()
                    browser.close()
                    return None
                
                # Wait for any redirects or dynamic content
                page.wait_for_timeout(2000)
                
                # Try to get PDF content via response body
                if response and response.ok:
                    try:
                        # Get response body from the navigation response
                        response_obj = page.wait_for_response(lambda r: r.url == meta.url or ".pdf" in r.url or "download" in r.url, timeout=10000)
                        if response_obj:
                            body_bytes = response_obj.body()
                            if body_bytes and body_bytes.startswith(b"%PDF"):
                                with open(local_path, "wb") as f:
                                    f.write(body_bytes)
                                logger.info(f"[browser] Saved PDF from navigation response: {local_path}")
                                context.close()
                                browser.close()
                                return local_path
                    except Exception as wait_err:
                        logger.warning(f"[browser] Could not get response body: {wait_err}")
                
                # Last resort: try to find download link on page
                content = page.content()
                if b"<html" in content.encode()[:1000]:
                    pdf_links = page.query_selector_all('a[href*=".pdf"], a[href*="download"], a[href*="legislations"]')
                    for link in pdf_links[:3]:  # Try first 3 links
                        href = link.get_attribute("href")
                        if href and (".pdf" in href.lower() or "download" in href.lower()):
                            actual_url = urljoin(meta.url, href)
                            logger.info(f"[browser] Trying link: {actual_url}")
                            try:
                                link_response = page.request.get(actual_url)
                                if link_response.ok:
                                    link_body = link_response.body()
                                    if link_body and link_body.startswith(b"%PDF"):
                                        with open(local_path, "wb") as f:
                                            f.write(link_body)
                                        logger.info(f"[browser] Saved PDF from link: {local_path}")
                                        context.close()
                                        browser.close()
                                        return local_path
                            except Exception:
                                continue
                
                logger.warning(f"[browser] Could not download PDF from {meta.url}")
                context.close()
                browser.close()
                return None
                
            except Exception as e:
                logger.error(f"[browser] Error during download: {e}")
                context.close()
                browser.close()
                return None
                
    except ImportError:
        logger.warning("Playwright not available for browser-based download. Install with: pip install playwright && playwright install chromium")
        return None
    except Exception as e:
        logger.error(f"Browser download failed for {meta.url}: {e}")
        return None


def _split_text(text: str, chunk_chars: int = 2000, overlap: int = 300) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


ARTICLE_PATTERN = re.compile(r"المادة\s*\d+", re.UNICODE)


def pdf_to_chunks(pdf_path: str, meta: PDFMeta) -> List[Chunk]:
    """Extract text from PDF (PyMuPDF if available, otherwise pypdf) and create chunks."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    page_texts: List[Tuple[int, str]] = []
    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                t = doc.load_page(i).get_text("text")
                page_texts.append((i + 1, t))
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed, falling back to pypdf: {e}")
            page_texts = _extract_with_pypdf(pdf_path)
    else:
        page_texts = _extract_with_pypdf(pdf_path)

    full_text = "\n".join([t for _, t in page_texts])
    raw_chunks = _split_text(full_text, chunk_chars=2000, overlap=300)

    # Heuristic: find article labels within each chunk
    result: List[Chunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        article_match = ARTICLE_PATTERN.search(chunk_text)
        article = article_match.group(0) if article_match else None
        pages_covered = _estimate_pages_for_chunk(chunk_text, page_texts)
        metadata = {
            "url": meta.url,
            "title": meta.title,
            "year": meta.year,
            "law_number": meta.law_number,
            "pdf_path": pdf_path,
            "article": article,
            "pages": ",".join(str(p) for p in pages_covered) if pages_covered else None,
            "lang": "ar",
            "sector": 47,
        }
        cid = _sha256_text((meta.url or "") + str(idx) + chunk_text[:32])
        result.append(Chunk(id=cid, text=chunk_text, metadata=metadata))

    logger.info(f"Created {len(result)} chunks from {os.path.basename(pdf_path)}")
    return result


def _extract_with_pypdf(pdf_path: str) -> List[Tuple[int, str]]:
    texts: List[Tuple[int, str]] = []
    try:
        reader = PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages, 1):
            try:
                texts.append((idx, page.extract_text() or ""))
            except Exception:
                texts.append((idx, ""))
    except Exception as e:
        logger.error(f"pypdf failed to read {pdf_path}: {e}")
    return texts


def _estimate_pages_for_chunk(chunk_text: str, page_texts: List[Tuple[int, str]]) -> List[int]:
    # naive: match presence across pages
    pages = []
    lowered = chunk_text.strip().split("\n")[0:3]
    probes = [l for l in lowered if l.strip()][:2]
    for pnum, ptext in page_texts:
        if any(probe and probe in ptext for probe in probes):
            pages.append(pnum)
    return pages[:5]


def _safe_filename(name: str) -> str:
    base = re.sub(r"[^\w\-\.\s\u0600-\u06FF]", "", name).strip()
    base = re.sub(r"\s+", " ", base).replace(" ", "_")
    return base or "document"


__all__ = [
    "PDFMeta",
    "Chunk",
    "discover_pdfs",
    "download_pdf",
    "pdf_to_chunks",
    "ensure_dir",
    "logger",
]


