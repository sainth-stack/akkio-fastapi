"""Row counts and caps for MVP / demo source data."""
from __future__ import annotations

# Fresh seed when app has no user upload
DEFAULT_SOURCE_SEED_ROWS = 12

# Excel template download (blank + sample rows only — never full production dump)
TEMPLATE_SAMPLE_ROWS = 12

# UI tables derived from source
MAX_TABLE_DISPLAY_ROWS = 24

# Auto-generated source rows (trim legacy huge mocks unless user uploaded)
MAX_AUTO_SOURCE_ROWS = 24
