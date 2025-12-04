"""
URL Web Scraping and Processing Module
Handles URL scraping, content extraction, and vector DB storage
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import Chroma
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    from langchain_community.embeddings import OpenAIEmbeddings
from database import PostgresDatabase
from datetime import datetime
import hashlib
import re
from urllib.parse import urlparse, urljoin

url_router = APIRouter()
db = PostgresDatabase()


def _clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
    return text.strip()


def _extract_text_from_url(url: str, timeout: int = 30) -> dict:
    """
    Scrape and extract content from a URL
    Returns dict with title, text content, links, and metadata
    """
    try:
        print(f"[URL_SCRAPER] Starting scrape for: {url}")
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Fetch the URL
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        print(f"[URL_SCRAPER] Successfully fetched URL, status: {response.status_code}")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else urlparse(url).netloc
        print(f"[URL_SCRAPER] Page title: {title_text}")
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Extract main content
        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|article', re.I))
        
        if main_content:
            content_soup = main_content
        else:
            content_soup = soup.find('body') or soup
        
        # Extract all text paragraphs
        paragraphs = []
        for p in content_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
            text = _clean_text(p.get_text())
            if text and len(text) > 20:  # Filter out very short paragraphs
                paragraphs.append(text)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            link_text = _clean_text(link.get_text())
            if href and not href.startswith(('#', 'javascript:')):
                absolute_url = urljoin(url, href)
                if link_text:
                    links.append({'url': absolute_url, 'text': link_text})
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            img_src = img.get('src', '')
            img_alt = img.get('alt', '')
            if img_src:
                absolute_img_url = urljoin(url, img_src)
                images.append({'url': absolute_img_url, 'alt': img_alt})
        
        # Combine all text content
        full_text = '\n\n'.join(paragraphs)
        
        print(f"[URL_SCRAPER] Extracted {len(paragraphs)} paragraphs, {len(links)} links, {len(images)} images")
        print(f"[URL_SCRAPER] Total text length: {len(full_text)} characters")
        
        return {
            'url': url,
            'title': title_text,
            'description': description,
            'text_content': full_text,
            'paragraphs': paragraphs,
            'links': links[:50],  # Limit to first 50 links
            'images': images[:20],  # Limit to first 20 images
            'word_count': len(full_text.split()),
            'paragraph_count': len(paragraphs),
            'scraped_at': datetime.now().isoformat()
        }
        
    except requests.Timeout:
        raise HTTPException(status_code=408, detail=f"Request timeout while fetching URL: {url}")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Connection error while fetching URL: {url}")
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error while fetching URL: {str(e)}")
    except Exception as e:
        print(f"[URL_SCRAPER] Error scraping URL: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better vector search"""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap  # Overlap to maintain context
    
    print(f"[URL_SCRAPER] Chunked text into {len(chunks)} chunks")
    return chunks


def _url_to_safe_name(url: str) -> str:
    """Convert URL to a safe filename/identifier"""
    # Create a hash of the URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Extract domain
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    
    # Clean domain for safe filename
    safe_domain = re.sub(r'[^\w\-]', '_', domain)
    
    return f"url_{safe_domain}_{url_hash}"


def _collection_name(email: str, url_name: str) -> str:
    """Generate collection name for ChromaDB"""
    safe_email = "".join(ch if ch.isalnum() else "_" for ch in (email or "user"))
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in (url_name or "url"))
    return f"{safe_email}__{safe_name}"


def _upsert_to_chromadb(email: str, url_name: str, texts: List[str], metadata: dict):
    """Store text chunks in ChromaDB for vector search"""
    coll_name = _collection_name(email, url_name)
    
    try:
        print(f"[URL_SCRAPER][VECTOR] Start upsert to Chroma: collection='{coll_name}', texts={len(texts)}")
    except Exception:
        pass
    
    if not texts:
        print(f"[URL_SCRAPER][VECTOR] No texts to index for collection='{coll_name}'. Skipping.")
        return
    
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        print(f"[URL_SCRAPER][VECTOR] OpenAIEmbeddings unavailable: {e}")
        return
    
    # Use same path as explore_api.py (2 levels up from api/akkio/)
    persist_dir = str(Path(__file__).resolve().parents[2] / "chroma_store")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        vectordb = Chroma(collection_name=coll_name, persist_directory=persist_dir, embedding_function=embeddings)
        
        # Delete existing collection to avoid duplicates
        try:
            existing_count = vectordb._collection.count()
            if existing_count > 0:
                print(f"[URL_SCRAPER][VECTOR] Deleting {existing_count} existing items in collection='{coll_name}'")
                vectordb.delete_collection()
                vectordb = Chroma(collection_name=coll_name, persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            print(f"[URL_SCRAPER][VECTOR] Warning during collection cleanup: {e}")
        
        ids = [f"{coll_name}_{i}" for i in range(len(texts))]
        vectordb.add_texts(texts=texts, metadatas=[metadata] * len(texts), ids=ids)
        
        print(f"[URL_SCRAPER][VECTOR] ✓ Upserted {len(texts)} chunks into collection='{coll_name}'")
    except Exception as e:
        print(f"[URL_SCRAPER][VECTOR] Error during upsert: {e}")
        import traceback
        traceback.print_exc()


@url_router.post("/api/process_url")
async def process_url(
    url: str = Form(...),
    mail: str = Form(...)
):
    """
    Scrape URL, extract content, store in database and vector store
    """
    try:
        print(f"[URL_SCRAPER] ═══════════════════════════════════════════════")
        print(f"[URL_SCRAPER] Processing URL: {url}")
        print(f"[URL_SCRAPER] User email: {mail}")
        print(f"[URL_SCRAPER] ═══════════════════════════════════════════════")
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")
        
        # Scrape and extract content
        print(f"[URL_SCRAPER] Step 1/4: Scraping URL content...")
        scraped_data = _extract_text_from_url(url)
        
        if not scraped_data['text_content'].strip():
            # Check for common blocking indicators
            title_lower = scraped_data.get('title', '').lower()
            if 'recaptcha' in title_lower or 'checking your browser' in title_lower:
                raise HTTPException(
                    status_code=400, 
                    detail="This website uses bot protection (reCAPTCHA). Please try a different URL. Wikipedia, documentation sites, and blogs work well."
                )
            raise HTTPException(status_code=400, detail="No text content could be extracted from the URL. The site may require JavaScript or have anti-scraping protection.")
        
        # Create safe name for storage
        url_name = _url_to_safe_name(url)
        print(f"[URL_SCRAPER] Generated safe name: {url_name}")
        
        # Create DataFrame from scraped content
        print(f"[URL_SCRAPER] Step 2/4: Creating DataFrame...")
        df_data = {
            'url': [url],
            'title': [scraped_data['title']],
            'description': [scraped_data['description']],
            'text_content': [scraped_data['text_content']],
            'word_count': [scraped_data['word_count']],
            'paragraph_count': [scraped_data['paragraph_count']],
            'scraped_at': [scraped_data['scraped_at']]
        }
        
        df = pd.DataFrame(df_data)
        print(f"[URL_SCRAPER] ✓ DataFrame created: {len(df)} rows × {len(df.columns)} columns")
        
        # Store in database
        print(f"[URL_SCRAPER] Step 3/4: Storing in database...")
        db_result = db.insert_or_update(
            email=mail,
            data=df,
            tb_name=url_name,
            data_type='url',
            data_subtype=None,
            raw_bytes=None
        )
        print(f"[URL_SCRAPER] ✓ Database storage complete: {db_result}")
        
        # Store metadata in a separate table for URL-specific info
        try:
            url_metadata_df = pd.DataFrame({
                'url_name': [url_name],
                'original_url': [url],
                'title': [scraped_data['title']],
                'description': [scraped_data['description']],
                'word_count': [scraped_data['word_count']],
                'scraped_at': [scraped_data['scraped_at']]
            })
            db.insert_or_update(
                email=mail,
                data=url_metadata_df,
                tb_name=f"{url_name}_metadata",
                data_type='url_metadata',
                data_subtype=None,
                raw_bytes=None
            )
        except Exception as e:
            print(f"[URL_SCRAPER] Warning: Could not store URL metadata: {e}")
        
        # Chunk text and store in vector database
        print(f"[URL_SCRAPER] Step 4/4: Storing in vector database...")
        chunks = _chunk_text(scraped_data['text_content'], chunk_size=500, overlap=50)
        
        # Add paragraph chunks for better context
        for paragraph in scraped_data['paragraphs'][:100]:  # Limit to first 100 paragraphs
            if len(paragraph) > 100:  # Only add substantial paragraphs
                chunks.append(paragraph)
        
        # Deduplicate chunks
        chunks = list(set(chunks))
        print(f"[URL_SCRAPER] Prepared {len(chunks)} unique chunks for vector storage")
        
        vector_metadata = {
            "email": mail,
            "name": url_name,
            "type": "url",
            "url": url,
            "title": scraped_data['title'],
            "scraped_at": scraped_data['scraped_at']
        }
        
        _upsert_to_chromadb(
            email=mail,
            url_name=url_name,
            texts=chunks,
            metadata=vector_metadata
        )
        
        print(f"[URL_SCRAPER] ═══════════════════════════════════════════════")
        print(f"[URL_SCRAPER] ✓✓✓ URL PROCESSING COMPLETE ✓✓✓")
        print(f"[URL_SCRAPER] URL: {url}")
        print(f"[URL_SCRAPER] Title: {scraped_data['title']}")
        print(f"[URL_SCRAPER] Word count: {scraped_data['word_count']}")
        print(f"[URL_SCRAPER] Chunks stored: {len(chunks)}")
        print(f"[URL_SCRAPER] ═══════════════════════════════════════════════")
        
        return JSONResponse(content={
            "message": "URL processed and stored successfully",
            "url_name": url_name,
            "original_url": url,
            "title": scraped_data['title'],
            "description": scraped_data['description'],
            "word_count": scraped_data['word_count'],
            "paragraph_count": scraped_data['paragraph_count'],
            "chunks_stored": len(chunks),
            "type": "url",
            "scraped_at": scraped_data['scraped_at']
        }, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[URL_SCRAPER] ✗ Error processing URL: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")


@url_router.post("/api/get_user_urls")
async def get_user_urls(email: str = Form(...)):
    """
    Get all URLs processed by a user
    """
    try:
        print(f"[URL_SCRAPER] Fetching URLs for user: {email}")
        
        # Get all user data from database
        all_data = db.get_user_data(email)
        
        if not all_data or not isinstance(all_data, list):
            return JSONResponse(content={"urls": []}, status_code=200)
        
        # Filter for URL type entries
        urls = []
        for item in all_data:
            if isinstance(item, dict) and item.get('type') == 'url':
                # Try to get metadata for this URL
                url_name = item.get('name', '')
                try:
                    metadata_df = db.get_table_data(f"{url_name}_metadata")
                    if metadata_df is not None and not metadata_df.empty:
                        metadata_row = metadata_df.iloc[0]
                        item['original_url'] = metadata_row.get('original_url', '')
                        item['title'] = metadata_row.get('title', '')
                        item['description'] = metadata_row.get('description', '')
                        item['word_count'] = metadata_row.get('word_count', 0)
                        item['scraped_at'] = metadata_row.get('scraped_at', '')
                except Exception as e:
                    print(f"[URL_SCRAPER] Could not fetch metadata for {url_name}: {e}")
                
                urls.append(item)
        
        print(f"[URL_SCRAPER] Found {len(urls)} URLs for user {email}")
        
        return JSONResponse(content={"urls": urls}, status_code=200)
        
    except Exception as e:
        print(f"[URL_SCRAPER] Error fetching user URLs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching user URLs: {str(e)}")

