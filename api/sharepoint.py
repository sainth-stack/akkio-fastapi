import os
import threading
from datetime import datetime, timezone
import requests
from urllib.parse import quote
import re
from pathlib import Path

# Load from environment, fallback to corrected values from .env
SHAREPOINT_CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID")
SHAREPOINT_CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET")
SHAREPOINT_TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID")
SHAREPOINT_SITE_URL = os.getenv("SHAREPOINT_SITE_URL")
# Default to folders at the drive root. You can override with full paths via env.
# IMPORTANT: Paths here are relative to the default document library drive root
# (which corresponds to "Shared Documents" in the SharePoint UI). Do not prefix
# these with "/Shared Documents" when using Graph drive endpoints.
SHAREPOINT_INPUT_FOLDER = os.getenv(
    "SHAREPOINT_INPUT_FOLDER",
    "/CH_SELE_ProEn/EL_SLA/SLA_Input_File/Pwr_Auto_DL",
)
SHAREPOINT_OUTPUT_FOLDER = os.getenv(
    "SHAREPOINT_OUTPUT_FOLDER",
    "/CH_SELE_ProEn/EL_SLA/SLA_Output_File",
)

# Automation interval configuration (minutes)
# Default to 1 minute for testing; override via env SHAREPOINT_AUTOMATION_MINUTES.
AUTOMATION_DEFAULT_MINUTES = int(os.getenv("SHAREPOINT_AUTOMATION_MINUTES", "1"))

# Reuse a single HTTP session for all Graph calls to reduce TLS/setup overhead
_SESSION = requests.Session()
_REQUEST_TIMEOUT = (5, 30)  # (connect, read) seconds - reduced for faster failure detection

# Cache Graph token and drive resolution to avoid repeated calls
_CACHED_ACCESS_TOKEN: str | None = None
_TOKEN_EXPIRES_AT: float | None = None
_CACHED_SITE: dict | None = None
_CACHED_DRIVE: dict | None = None

# Legacy persistence functions removed - now using main API's enhanced de-duplication

def get_app_token() -> str:
    """Acquire app-only token for Microsoft Graph with simple in-process caching."""
    global _CACHED_ACCESS_TOKEN, _TOKEN_EXPIRES_AT
    from time import time as _now
    if _CACHED_ACCESS_TOKEN and _TOKEN_EXPIRES_AT and _now() < _TOKEN_EXPIRES_AT - 60:
        return _CACHED_ACCESS_TOKEN

    token_url = f"https://login.microsoftonline.com/{SHAREPOINT_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": SHAREPOINT_CLIENT_ID,
        "client_secret": SHAREPOINT_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    print(f"🔄 Requesting token from: {token_url}")
    resp = _SESSION.post(token_url, data=data, timeout=_REQUEST_TIMEOUT)
    print(f"Token response status: {resp.status_code}")
    if not resp.ok:
        print(f"❌ Token request failed: {resp.text}")
        raise RuntimeError(f"Token request failed: {resp.text}")
    payload = resp.json()
    _CACHED_ACCESS_TOKEN = payload.get("access_token")
    # Default token lifetime ~3600s. Respect expires_in if present.
    try:
        ttl = int(payload.get("expires_in", 3600))
    except Exception:
        ttl = 3600
    _TOKEN_EXPIRES_AT = _now() + ttl
    print("✅ Access token obtained successfully")
    return _CACHED_ACCESS_TOKEN

def graph_get(url: str, token: str) -> dict:
    """Make an authenticated GET request to Microsoft Graph API."""
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🔄 Making Graph API request to: {url}")
    res = _SESSION.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
    print(f"Graph API response status: {res.status_code}")
    
    if not res.ok:
        print(f"❌ Graph API error: {res.text}")
        raise RuntimeError(f"Graph API error: {res.text}")
    
    return res.json()

def graph_get_binary(url: str, token: str) -> bytes:
    """Make an authenticated GET request to Microsoft Graph API that returns raw bytes (e.g., file content)."""
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🔄 Downloading binary content from: {url}")
    res = _SESSION.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
    print(f"Binary response status: {res.status_code}")
    if not res.ok:
        print(f"❌ Binary download error: {res.text}")
        raise RuntimeError(f"Binary download error: {res.text}")
    return res.content

def resolve_site_and_drive(token: str):
    """Resolve site and drive information from SharePoint."""
    global _CACHED_SITE, _CACHED_DRIVE
    if _CACHED_SITE and _CACHED_DRIVE:
        return _CACHED_SITE, _CACHED_DRIVE
    # Use the same format as working JS: host:/path
    site_url = SHAREPOINT_SITE_URL or ''
    # Accept full https URL or host:/path format
    if site_url.startswith("http://") or site_url.startswith("https://"):
        # strip scheme
        without_scheme = site_url.split('://', 1)[1]
        host, path = without_scheme.split('/', 1)
        site_endpoint = f"https://graph.microsoft.com/v1.0/sites/{host}:/{path}"
    elif '/' in site_url:
        host, path = site_url.split('/', 1)
        site_endpoint = f"https://graph.microsoft.com/v1.0/sites/{host}:/{path}"
    else:
        site_endpoint = f"https://graph.microsoft.com/v1.0/sites/{site_url}"

    print(f"🔄 Resolving site from: {site_endpoint}")
    site = graph_get(site_endpoint, token)
    print(f"✅ Site found: {site.get('displayName', 'Unknown')}")
    
    drive_endpoint = f"https://graph.microsoft.com/v1.0/sites/{site['id']}/drive"
    drive = graph_get(drive_endpoint, token)
    print(f"✅ Drive found: {drive.get('name', 'Unknown')}")
    _CACHED_SITE, _CACHED_DRIVE = site, drive
    return _CACHED_SITE, _CACHED_DRIVE

def list_folder_children(token: str, drive_id: str, folder_path: str) -> dict:
    """List files inside a given SharePoint folder path."""
    encoded = quote(folder_path, safe="")
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}:/children"
    print(f"🔄 Listing folder contents: {folder_path}")
    return graph_get(url, token)

def try_get_item_by_path(token: str, drive_id: str, item_path: str) -> dict | None:
    """Fast existence check by item path. Returns item JSON if exists else None.

    This call should be fast and never block the main upload flow. On any error,
    we log and return None so the caller proceeds to upload.
    """
    encoded = quote(item_path, safe="")
    # Select minimal fields to speed up response
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}?$select=id,name" 
    headers = {"Authorization": f"Bearer {token}"}
    try:
        print(f"🔎 Checking item existence by path: {item_path}")
        res = _SESSION.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        print(f"🔎 Existence check status: {res.status_code}")
        if res.status_code == 200:
            return res.json()
        if res.status_code == 404:
            return None
        # For other statuses, treat as not-found to avoid blocking
        print(f"ℹ️ Non-200/404 on existence check: {res.status_code} {res.text[:200]}")
        return None
    except Exception as e:
        print(f"⚠️ Existence check failed ({type(e).__name__}): {e}")
        return None

def ensure_folder_exists(token: str, drive_id: str, folder_path: str) -> bool:
    """Ensure a folder exists at the given path relative to drive root.

    Only supports a single-level root folder like "/FolderName". For nested paths,
    set env vars accordingly or create them manually.
    """
    try:
        # Try to fetch the folder
        encoded = quote(folder_path, safe="")
        folder_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}"
        print(f"🔄 Checking/ensuring folder exists at: {folder_path}")
        graph_get(folder_url, token)
        return True
    except RuntimeError as e:
        if "itemNotFound" in str(e):
            # Create only the last segment at root
            folder_name = folder_path.strip("/")
            print(f"🔄 Creating root-level folder: {folder_name}")
            create_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            data = {"name": folder_name, "folder": {}}
            response = _SESSION.post(create_url, json=data, headers=headers, timeout=_REQUEST_TIMEOUT)
            if response.ok:
                print(f"✅ Folder created successfully: {folder_name}")
                return True
            print(f"❌ Failed to create folder {folder_name}: {response.text}")
            return False
        print(f"❌ Error ensuring folder {folder_path}: {e}")
        return False

def upload_file_to_folder(
    token: str,
    drive_id: str,
    folder_path: str,
    file_name: str,
    file_bytes: bytes,
) -> dict:
    """Upload file to SharePoint output folder, using chunked session for large files.

    - Uses simple PUT for files <= 4 MB
    - Uses Microsoft Graph upload session with chunked upload for larger files
    """
    total_size = len(file_bytes)
    print(f"🔄 Starting upload process for: {file_name} ({total_size} bytes)")

    print(f"🔄 Ensuring folder exists: {folder_path}")
    if not ensure_folder_exists(token, drive_id, folder_path):
        raise RuntimeError(f"Destination folder does not exist and could not be created: {folder_path}")
    print(f"✅ Folder confirmed: {folder_path}")

    # Route based on size
    SIMPLE_UPLOAD_MAX = 4 * 1024 * 1024  # 4 MB
    if total_size <= SIMPLE_UPLOAD_MAX:
        dest_path = f"{folder_path.rstrip('/')}/{file_name}"
        encoded = quote(dest_path, safe="")
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}:/content"
        headers = {"Authorization": f"Bearer {token}"}

        print(f"🔄 Uploading (simple) to: {dest_path}")
        print(f"🔄 Upload URL: {url}")
        print(f"🔄 Starting PUT request with {total_size} bytes...")

        try:
            upload_timeout = (10, 120)  # 10s connect, 2min read timeout
            print(f"🔄 Using timeout: {upload_timeout}")

            resp = _SESSION.put(url, headers=headers, data=file_bytes, timeout=upload_timeout)
            print(f"🔄 PUT request completed. Status: {resp.status_code}")

            if not resp.ok:
                error_text = resp.text[:500]
                print(f"❌ Upload failed: {resp.status_code} {error_text}")
                raise RuntimeError(f"Upload failed [{resp.status_code}]: {error_text}")

            result = resp.json()
            print("✅ Upload successful")
            print(f"✅ Uploaded item ID: {result.get('id', 'unknown')}")
            # Log the uploaded item's final name as stored in SharePoint
            uploaded_name = result.get('name') or file_name
            print(f"📄 Uploaded item name: {uploaded_name}")
            return result

        except requests.exceptions.Timeout as e:
            print(f"❌ Upload timed out: {e}")
            raise RuntimeError(f"Upload timed out - file may be too large or connection is slow: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error during upload: {e}")
            raise RuntimeError(f"Connection error during upload: {e}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request exception during upload: {e}")
            raise RuntimeError(f"Request failed during upload: {e}")
        except Exception as e:
            print(f"❌ Unexpected exception during upload: {type(e).__name__}: {e}")
            raise RuntimeError(f"Unexpected error during upload: {e}")

    # Large file path -> chunked upload session
    dest_path = f"{folder_path.rstrip('/')}/{file_name}"
    encoded = quote(dest_path, safe="")
    session_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}:/createUploadSession"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    print(f"🔄 Creating upload session for: {dest_path}")
    session_body = {
        "item": {
            "@microsoft.graph.conflictBehavior": "fail"
        }
    }

    try:
        session_resp = _SESSION.post(session_url, json=session_body, headers=headers, timeout=(10, 60))
        if not session_resp.ok:
            print(f"❌ Failed to create upload session: {session_resp.status_code} {session_resp.text[:300]}")
            raise RuntimeError(f"CreateUploadSession failed [{session_resp.status_code}]: {session_resp.text}")
        upload_url = session_resp.json().get("uploadUrl")
        if not upload_url:
            raise RuntimeError("Upload session URL missing in response")
        print("✅ Upload session created")

        # Chunked upload
        # Start conservatively to avoid proxy/network write timeouts
        chunk_size = 1 * 1024 * 1024  # 1 MB initial chunk size
        min_chunk_size = 256 * 1024   # 256 KB minimum
        bytes_sent = 0
        attempt = 0
        max_attempts_per_size = 3

        while bytes_sent < total_size:
            start = bytes_sent
            end = min(start + chunk_size, total_size) - 1
            chunk = file_bytes[start:end + 1]
            content_length = end - start + 1
            content_range = f"bytes {start}-{end}/{total_size}"
            # Per Graph docs, uploadUrl is pre-authorized; no Authorization header needed
            chunk_headers = {
                "Content-Length": str(content_length),
                "Content-Range": content_range,
            }

            try:
                print(f"🔄 Uploading chunk {start}-{end} ({content_length} bytes)")
                # Longer read timeout for uploading chunks (slow networks)
                chunk_timeout = (20, 600)
                resp = _SESSION.put(upload_url, headers=chunk_headers, data=chunk, timeout=chunk_timeout)

                # 202 indicates more chunks to send; 200/201 indicates completion
                if resp.status_code in (200, 201):
                    result = resp.json()
                    print("✅ Chunked upload completed successfully")
                    print(f"✅ Uploaded item ID: {result.get('id', 'unknown')}")
                    # Log the uploaded item's final name as stored in SharePoint
                    uploaded_name = result.get('name') or file_name
                    print(f"📄 Uploaded item name: {uploaded_name}")
                    return result
                if resp.status_code == 202:
                    bytes_sent = end + 1
                    attempt = 0  # reset attempt counter on success
                    continue

                # Other statuses -> error
                print(f"❌ Chunk upload failed: {resp.status_code} {resp.text[:300]}")
                raise RuntimeError(f"Chunk upload failed [{resp.status_code}]: {resp.text}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt += 1
                print(f"⚠️ Chunk upload error (attempt {attempt}/{max_attempts_per_size}) with chunk_size={chunk_size} bytes: {e}")
                if attempt >= max_attempts_per_size:
                    # Reduce chunk size and retry from same position
                    new_chunk_size = max(min_chunk_size, chunk_size // 2)
                    if new_chunk_size < chunk_size:
                        print(f"↘️ Reducing chunk size from {chunk_size} to {new_chunk_size} bytes and retrying")
                        chunk_size = new_chunk_size
                        attempt = 0
                    else:
                        raise RuntimeError(f"Chunk upload failed after retries at minimum chunk size: {e}")
                # Backoff before retry
                import time as _time
                _time.sleep(1.5 * attempt)
                continue

        # If loop exits without return, something went wrong
        raise RuntimeError("Upload session ended unexpectedly without completion")

    except Exception as e:
        print(f"❌ Upload session error: {e}")
        raise


def ensure_folder_exists(token: str, drive_id: str, folder_path: str) -> bool:
    """Ensure a folder exists at the given path relative to drive root.

    Only supports a single-level root folder like "/FolderName". For nested paths,
    set env vars accordingly or create them manually.
    """
    try:
        # Try to fetch the folder
        encoded = quote(folder_path, safe="")
        folder_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}"
        print(f"🔄 Checking/ensuring folder exists at: {folder_path}")
        
        # Add timeout to folder check
        headers = {"Authorization": f"Bearer {token}"}
        resp = _SESSION.get(folder_url, headers=headers, timeout=_REQUEST_TIMEOUT)
        
        if resp.ok:
            print(f"✅ Folder exists: {folder_path}")
            return True
        elif resp.status_code == 404:
            # Folder doesn't exist, try to create it
            print(f"📁 Folder not found, attempting to create: {folder_path}")
        else:
            print(f"⚠️ Unexpected response when checking folder: {resp.status_code} {resp.text[:200]}")
            return False
            
    except Exception as e:
        print(f"⚠️ Error checking folder existence: {e}")
        # Continue to try creating the folder
    
    # Create the folder
    try:
        folder_name = folder_path.strip("/")
        if "/" in folder_name:
            print(f"❌ Nested folder creation not supported: {folder_path}")
            return False
            
        print(f"🔄 Creating root-level folder: {folder_name}")
        create_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        data = {"name": folder_name, "folder": {}}
        
        response = _SESSION.post(create_url, json=data, headers=headers, timeout=_REQUEST_TIMEOUT)
        
        if response.ok:
            print(f"✅ Folder created successfully: {folder_name}")
            return True
        elif response.status_code == 409:
            # Folder already exists (race condition)
            print(f"ℹ️ Folder already exists (created concurrently): {folder_name}")
            return True
        else:
            print(f"❌ Failed to create folder {folder_name}: {response.status_code} {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Exception creating folder {folder_path}: {e}")
        return False
        
def download_file_from_path(token: str, drive_id: str, file_path: str) -> bytes:
    """Download file bytes from a drive using a root-relative path (e.g., /Folder/Sub/file.xlsx)."""
    encoded = quote(file_path, safe="")
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}:/content"
    return graph_get_binary(url, token)

def delete_drive_item(token: str, drive_id: str, item_id: str) -> bool:
    """Delete an item in the drive by its ID. Returns True on success."""
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🗑️ Deleting drive item: {item_id}")
    resp = _SESSION.delete(url, headers=headers, timeout=_REQUEST_TIMEOUT)
    print(f"Delete response status: {resp.status_code}")
    if resp.status_code in (200, 204):
        print("✅ Delete successful")
        return True
    print(f"❌ Delete failed: {resp.text}")
    raise RuntimeError(f"Delete failed: {resp.text}")

def select_latest_file(items: list) -> dict | None:
    """Select the latest-modified file item from a list of drive children."""
    file_items = [it for it in (items or []) if isinstance(it, dict) and it.get("file") is not None]
    if not file_items:
        return None
    def parse_dt(s: str) -> datetime:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    latest = max(file_items, key=lambda it: parse_dt(it.get("lastModifiedDateTime", "")))
    return latest

def simple_process_to_report(input_bytes: bytes, source_name: str) -> tuple[str, bytes]:
    """Return bytes unchanged but name output after input file with date-time.

    Example: input "data_file.xlsx" -> "data_file_20250131_143015.xlsx"
    """
    # Derive a safe base name from the source file (no extension, sanitized)
    base = os.path.splitext(os.path.basename(source_name or "report"))[0]
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._-") or "report"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{safe_base}_{timestamp}.xlsx"
    return out_name, input_bytes

_automation_thread: threading.Thread | None = None
_automation_stop = threading.Event()
_automation_interval_seconds: int | None = None
_last_processed_item_id: str | None = None
_last_run_time: datetime | None = None
_last_error_message: str | None = None

def run_automation_cycle():
    """Run one automation cycle: find latest input, download, process, upload to output."""
    global _last_processed_item_id
    print("\n🕒 Running SharePoint automation cycle...")
    global _last_run_time, _last_error_message
    token = get_app_token()
    _, drive = resolve_site_and_drive(token)
    drive_id = drive["id"]

    # List input folder
    children = list_folder_children(token, drive_id, SHAREPOINT_INPUT_FOLDER)
    items = children.get("value", [])
    print(f"📁 Found {len(items)} items in input folder")
    # Consider only HDA files
    hda_items = [it for it in items if isinstance(it, dict) and it.get("file") is not None and (it.get("name") or "").lower().find("hda") != -1]
    latest = select_latest_file(hda_items)
    if not latest:
        print("ℹ️ No HDA files found to process.")
        return

    item_id = latest.get("id")
    item_name = latest.get("name")
    parent_path = (latest.get("parentReference", {}) or {}).get("path", "")
    parent_root_prefix = f"/drives/{drive_id}/root:"
    if parent_path.startswith(parent_root_prefix):
        parent_rel = parent_path[len(parent_root_prefix):]
    else:
        parent_rel = SHAREPOINT_INPUT_FOLDER
    file_path = f"{parent_rel}/{item_name}"

    # Process only files whose names contain 'HDA' or 'hda'
    try:
        if not (isinstance(item_name, str) and ("hda" in item_name.lower())):
            print(f"⏭️ Skipping file that does not match HDA filter: {item_name}")
            return
    except Exception:
        print("⏭️ Skipping file due to invalid name for HDA check")
        return

    if _last_processed_item_id == item_id:
        print(f"⏭️ Latest file already processed recently: {item_name}")
        return

    # De-dup: skip if same source input name was processed already (check via main API's cache)
    try:
        from .. import final_akio_apis
        if item_name and final_akio_apis._is_file_recently_processed(item_name):
            print(f"⏭️ Skipping processing; input '{item_name}' already processed recently.")
            return
    except ImportError:
        # Fallback to simple item ID check if main API not available
        pass

    print(f"⬇️ Downloading latest file: {file_path}")
    file_bytes = download_file_from_path(token, drive_id, file_path)

    print("🛠️ Processing file to report format...")
    out_name, out_bytes = simple_process_to_report(file_bytes, item_name)

    print(f"⬆️ Uploading report to output folder as: {out_name}")
    upload_file_to_folder(
        token=token,
        drive_id=drive_id,
        folder_path=SHAREPOINT_OUTPUT_FOLDER,
        file_name=out_name,
        file_bytes=out_bytes,
    )
    # Mark the input file as processed in the main API's cache
    try:
        from .. import final_akio_apis
        if item_name:
            final_akio_apis._mark_file_as_processed(item_name)
    except ImportError:
        # Fallback: no-op if main API not available
        pass
    _last_processed_item_id = item_id
    _last_run_time = datetime.now(timezone.utc)
    _last_error_message = None
    print("✅ Automation cycle completed")

def process_latest_if_new() -> dict:
    """Run one cycle (no thread) and return simple status.

    This lets an external caller (e.g., an API hit every ~10 minutes) trigger
    processing of the most recently uploaded input file and upload the processed
    report to the output folder, if a new file is detected.
    """
    try:
        run_automation_cycle()
    except Exception as e:
        # Surface the error in status while keeping the call simple
        global _last_error_message
        _last_error_message = str(e)
    return get_sharepoint_automation_status()

def _automation_loop(interval_seconds: int):
    global _last_error_message
    while not _automation_stop.is_set():
        try:
            run_automation_cycle()
        except Exception as e:
            _last_error_message = str(e)
            print(f"❌ Automation cycle error: {e}")
        finally:
            _automation_stop.wait(interval_seconds)

def start_sharepoint_automation(every_minutes: int = AUTOMATION_DEFAULT_MINUTES):
    """Start background automation to process latest input file and upload report periodically."""
    global _automation_thread
    if _automation_thread and _automation_thread.is_alive():
        print("ℹ️ SharePoint automation already running")
        return
    global _automation_interval_seconds
    interval = max(1, int(every_minutes)) * 60
    print(f"🚀 Starting SharePoint automation (every {every_minutes} minutes)")
    _automation_stop.clear()
    _automation_interval_seconds = interval
    _automation_thread = threading.Thread(
        target=_automation_loop, args=(interval,), daemon=True
    )
    _automation_thread.start()

def stop_sharepoint_automation():
    """Stop the background automation thread."""
    print("🛑 Stopping SharePoint automation...")
    _automation_stop.set()
    if _automation_thread:
        _automation_thread.join(timeout=5)

def run_sharepoint_automation_once():
    """Trigger a single automation cycle immediately."""
    run_automation_cycle()

def get_sharepoint_automation_status() -> dict:
    """Return current automation status and last run details."""
    running = bool(_automation_thread and _automation_thread.is_alive())
    return {
        "running": running,
        "interval_minutes": int(_automation_interval_seconds / 60) if _automation_interval_seconds else None,
        "last_processed_item_id": _last_processed_item_id,
        "last_run_time": _last_run_time.isoformat().replace("+00:00", "Z") if _last_run_time else None,
        "last_error": _last_error_message,
    }

def list_sharepoint_files(folder: str = None):
    """List files for a given SharePoint folder path."""
    # Only list files; do not trigger any background processing from here.

    token = get_app_token()
    _, drive = resolve_site_and_drive(token)
    drive_id = drive["id"]

    if folder:
        items = list_folder_children(token, drive_id, folder)
        return {"folder": folder, "items": items.get("value", [])}

    input_items = list_folder_children(token, drive_id, SHAREPOINT_INPUT_FOLDER).get("value", [])
    output_items = list_folder_children(token, drive_id, SHAREPOINT_OUTPUT_FOLDER).get("value", [])
    return {
        "input": {"folder": SHAREPOINT_INPUT_FOLDER, "items": input_items},
        "output": {"folder": SHAREPOINT_OUTPUT_FOLDER, "items": output_items}
    }