import os
import threading
import time
from datetime import datetime, timezone
import requests
from urllib.parse import quote
import re

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
    "/CH_SELE_ProEn/EL_SLA/SLA_Input_File",
)
SHAREPOINT_OUTPUT_FOLDER = os.getenv(
    "SHAREPOINT_OUTPUT_FOLDER",
    "/CH_SELE_ProEn/EL_SLA/SLA_Output_File",
)

def get_app_token() -> str:
    """Acquire app-only token for Microsoft Graph."""
    token_url = f"https://login.microsoftonline.com/{SHAREPOINT_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": SHAREPOINT_CLIENT_ID,
        "client_secret": SHAREPOINT_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    
    print(f"🔄 Requesting token from: {token_url}")
    print(f"Using Client ID: {SHAREPOINT_CLIENT_ID}")
    print(f"Using Tenant ID: {SHAREPOINT_TENANT_ID}")
    
    resp = requests.post(token_url, data=data)
    print(f"Token response status: {resp.status_code}")
    
    if not resp.ok:
        print(f"❌ Token request failed: {resp.text}")
        raise RuntimeError(f"Token request failed: {resp.text}")
    
    print("✅ Access token obtained successfully")
    return resp.json().get("access_token")

def graph_get(url: str, token: str) -> dict:
    """Make an authenticated GET request to Microsoft Graph API."""
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🔄 Making Graph API request to: {url}")
    
    res = requests.get(url, headers=headers)
    print(f"Graph API response status: {res.status_code}")
    
    if not res.ok:
        print(f"❌ Graph API error: {res.text}")
        raise RuntimeError(f"Graph API error: {res.text}")
    
    return res.json()

def graph_get_binary(url: str, token: str) -> bytes:
    """Make an authenticated GET request to Microsoft Graph API that returns raw bytes (e.g., file content)."""
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🔄 Downloading binary content from: {url}")
    res = requests.get(url, headers=headers)
    print(f"Binary response status: {res.status_code}")
    if not res.ok:
        print(f"❌ Binary download error: {res.text}")
        raise RuntimeError(f"Binary download error: {res.text}")
    return res.content

def resolve_site_and_drive(token: str):
    """Resolve site and drive information from SharePoint."""
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
    
    return site, drive

def list_folder_children(token: str, drive_id: str, folder_path: str) -> dict:
    """List files inside a given SharePoint folder path."""
    encoded = quote(folder_path, safe="")
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}:/children"
    print(f"🔄 Listing folder contents: {folder_path}")
    return graph_get(url, token)

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
            response = requests.post(create_url, json=data, headers=headers)
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
    """Upload a small file to the given folder in SharePoint using a simple PUT.

    For files larger than ~4MB, a chunked upload session should be used instead.
    """
    if not ensure_folder_exists(token, drive_id, folder_path):
        raise RuntimeError(f"Destination folder does not exist and could not be created: {folder_path}")

    dest_path = f"{folder_path.rstrip('/')}/{file_name}"
    encoded = quote(dest_path, safe="")
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{encoded}:/content"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    }
    print(f"🔄 Uploading to: {dest_path}")
    resp = requests.put(url, headers=headers, data=file_bytes)
    if not resp.ok:
        print(f"❌ Upload failed: {resp.status_code} {resp.text}")
        raise RuntimeError(f"Upload failed: {resp.text}")
    print("✅ Upload successful")
    return resp.json()

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
    resp = requests.delete(url, headers=headers)
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
    latest = select_latest_file(items)
    if not latest:
        print("ℹ️ No files found to process.")
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

    if _last_processed_item_id == item_id:
        print(f"⏭️ Latest file already processed recently: {item_name}")
        return

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

def start_sharepoint_automation(every_minutes: int = 10):
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
    # If caller is listing the configured input folder, also attempt a simple
    # process cycle so that hitting this endpoint periodically (e.g. every 10m)
    # both checks and processes the latest file when it changes.
    try:
        if folder and folder.strip() == SHAREPOINT_INPUT_FOLDER:
            process_latest_if_new()
    except Exception:
        # Non-fatal for listing; still attempt to list below
        pass

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