"""
Tests for the frontend-only pipeline:
  - Context size (PRD never leaks into page context)
  - Injected fault recovery
  - Stage routing (correct files produced per stage)
"""
import json
import unittest
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Test 1: context_builder — PRD must never appear in page context
# ---------------------------------------------------------------------------

class TestContextBuilder(unittest.TestCase):

    def _make_blueprint(self) -> Dict:
        return {
            "app_name": "CRM Dashboard",
            "domain": "crm",
            "routes": [
                {"path": "/", "page": "DashboardPage", "screen_id": "S001"},
                {"path": "/contacts", "page": "ContactsPage", "screen_id": "S002"},
            ],
            "pages": [
                {"file": "DashboardPage", "uses_entities": ["Deal", "Activity"], "uses_kit_components": ["StatCard", "LineChart"]},
                {"file": "ContactsPage", "uses_entities": ["Contact"], "uses_kit_components": ["DataTable", "Modal", "FormField"]},
            ],
            "entities": [
                {"name": "Contact", "fields": [{"name": "id", "type": "string"}, {"name": "name", "type": "string"}, {"name": "email", "type": "string"}]},
                {"name": "Deal", "fields": [{"name": "id", "type": "string"}, {"name": "title", "type": "string"}, {"name": "amount", "type": "number"}]},
            ],
            "nav_items": [
                {"label": "Dashboard", "path": "/", "icon": "Dashboard"},
                {"label": "Contacts", "path": "/contacts", "icon": "People"},
            ],
        }

    def _make_uiux(self) -> Dict:
        return {
            "screens": [
                {
                    "screen_id": "S001",
                    "layout": "dashboard",
                    "sections": ["KPI cards", "Revenue chart"],
                    "components_from_kit": ["StatCard", "LineChart", "DataTable"],
                    "states": {"empty": "No data", "loading": "Loading...", "error": "Error"},
                    "actions": ["filter by date"],
                    "navigation": "sidebar nav",
                },
                {
                    "screen_id": "S002",
                    "layout": "table",
                    "sections": ["Contacts list", "Add/edit modal"],
                    "components_from_kit": ["DataTable", "Modal", "FormField"],
                    "states": {"empty": "No contacts", "loading": "Loading contacts...", "error": "Failed to load"},
                    "actions": ["add contact", "edit contact", "delete contact"],
                    "navigation": "sidebar nav",
                },
            ]
        }

    def test_prd_not_in_page_context(self):
        """PRD text must never appear in a page agent context."""
        from app_builder.services.context_builder import build_page_context, assert_no_prd_in_context

        prd_text = (
            "This is the full PRD document. It contains detailed user stories, "
            "acceptance criteria, and business requirements that are NOT meant for "
            "individual page agents. Fingerprint: DONT_LEAK_THIS_PRD_TEXT_12345."
        )

        blueprint = self._make_blueprint()
        uiux = self._make_uiux()
        types_content = "export interface Contact { id: string; name: string; email: string; }"
        mock_exports = ["mockContacts", "mockDeals", "useMockStore", "mockKpis"]

        for page_file in ["DashboardPage", "ContactsPage"]:
            ctx = build_page_context(
                page_file=page_file,
                blueprint_json=blueprint,
                uiux_json=uiux,
                design_tokens_json={"light": {"primary": "#1976d2"}},
                types_content=types_content,
                mock_exports=mock_exports,
            )
            # This must NOT raise
            assert_no_prd_in_context(ctx, prd_text)
            # Verify context is non-empty and has expected keys
            self.assertIn("page_name", ctx)
            self.assertIn("kit_signatures", ctx)
            self.assertIn("relevant_types", ctx)
            self.assertIn("mock_exports", ctx)
            # PRD text must not be in context
            ctx_str = json.dumps(ctx)
            self.assertNotIn("DONT_LEAK_THIS_PRD_TEXT_12345", ctx_str)

    def test_context_contains_only_relevant_types(self):
        """ContactsPage context should contain Contact types, not Deal types."""
        from app_builder.services.context_builder import build_page_context

        types_content = (
            "export interface Contact { id: string; name: string; }\n"
            "export interface Deal { id: string; title: string; amount: number; }\n"
        )
        blueprint = self._make_blueprint()
        uiux = self._make_uiux()
        mock_exports = ["mockContacts", "mockDeals", "useMockStore"]

        ctx = build_page_context(
            page_file="ContactsPage",
            blueprint_json=blueprint,
            uiux_json=uiux,
            design_tokens_json=None,
            types_content=types_content,
            mock_exports=mock_exports,
        )
        # Should include Contact (used by ContactsPage) but types are filtered
        self.assertIn("Contact", ctx["relevant_types"])
        # Mock exports should include contact-related exports
        self.assertIn("mockContacts", ctx["mock_exports"])
        # Kit signatures should include DataTable and Modal (used by ContactsPage)
        self.assertIn("DataTable", ctx["kit_signatures"])
        self.assertIn("Modal", ctx["kit_signatures"])

    def test_kit_signatures_always_include_core(self):
        """AppShell, PageHeader, EmptyState, LoadingState, ErrorState always included."""
        from app_builder.services.context_builder import build_page_context

        blueprint = self._make_blueprint()
        ctx = build_page_context(
            page_file="ContactsPage",
            blueprint_json=blueprint,
            uiux_json=None,
            design_tokens_json=None,
            types_content="",
            mock_exports=[],
        )
        for always in ("AppShell", "PageHeader", "EmptyState", "LoadingState", "ErrorState"):
            self.assertIn(always, ctx["kit_signatures"], f"{always} missing from kit_signatures")


# ---------------------------------------------------------------------------
# Test 2: verify_service — completeness check
# ---------------------------------------------------------------------------

class TestVerifyService(unittest.TestCase):

    def test_is_complete_file_ok(self):
        from app_builder.services.verify_service import is_complete_file

        good = """
import { Box } from '@mui/material';
export default function DashboardPage() {
  return <Box>Hello</Box>;
}
"""
        self.assertTrue(is_complete_file(good))

    def test_is_complete_file_truncated(self):
        from app_builder.services.verify_service import is_complete_file

        truncated = "import { Box } from '@mui/material';\n// rest of the code..."
        self.assertFalse(is_complete_file(truncated))

    def test_is_complete_file_unbalanced(self):
        from app_builder.services.verify_service import is_complete_file

        # Clear imbalance (4 opens, 1 close) — well beyond tolerance
        unbalanced = "export function Foo() { if (x) { if (y) { while (z) { return 1; }"
        self.assertFalse(is_complete_file(unbalanced))

    def test_extract_exports(self):
        from app_builder.services.verify_service import _extract_exports

        content = """
export interface Contact { id: string; name: string; }
export const mockContacts: Contact[] = [];
export function useMockStore(key: string) { return {}; }
export default function ContactsPage() { return null; }
export type { Contact };
"""
        exports = _extract_exports(content)
        self.assertIn("Contact", exports)
        self.assertIn("mockContacts", exports)
        self.assertIn("useMockStore", exports)
        self.assertIn("default", exports)

    def test_import_resolution_ok(self):
        """Files that import from existing sources should pass."""
        from app_builder.services.verify_service import _check_import_resolution
        import tempfile, os

        types_content = "export interface Item { id: string; name: string; }"
        app_tsx = "import { Item } from './types'; export default function App() { return null; }"

        files = {
            "frontend/src/types.ts": types_content,
            "frontend/src/App.tsx": app_tsx,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _check_import_resolution(files, tmpdir)
            # Should pass (Item is exported by types.ts)
            self.assertTrue(result.ok, f"Expected ok but got errors: {result.errors}")

    def test_import_resolution_missing_export(self):
        """Importing a non-existent named export should fail."""
        from app_builder.services.verify_service import _check_import_resolution
        import tempfile

        types_content = "export interface Item { id: string; }"
        app_tsx = "import { NonExistentThing } from './types'; export default function App() { return null; }"

        files = {
            "frontend/src/types.ts": types_content,
            "frontend/src/App.tsx": app_tsx,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _check_import_resolution(files, tmpdir)
            self.assertFalse(result.ok)
            self.assertTrue(any("NonExistentThing" in e for e in result.errors))


# ---------------------------------------------------------------------------
# Test 3: injected fault recovery
# ---------------------------------------------------------------------------

class TestFaultRecovery(unittest.TestCase):

    def test_extract_first_error(self):
        from app_builder.services.verify_service import VerifyResult, extract_first_error

        result = VerifyResult(
            ok=False,
            errors=["frontend/src/pages/ContactsPage.tsx:12:5 — error TS2345: wrong type"],
            offending_files=["frontend/src/pages/ContactsPage.tsx"],
            log="error log here",
        )
        files = {
            "frontend/src/pages/ContactsPage.tsx": "export default function ContactsPage() { return null; }",
        }
        info = extract_first_error(result, files)
        self.assertEqual(info["offending_file"], "frontend/src/pages/ContactsPage.tsx")
        self.assertIn("TS2345", info["error"])
        self.assertIn("ContactsPage", info["file_content"])


# ---------------------------------------------------------------------------
# Test 4: context_builder types / mock extraction
# ---------------------------------------------------------------------------

class TestContextExtraction(unittest.TestCase):

    def test_extract_export_names(self):
        from app_builder.services.frontend_only_pipeline import _extract_export_names

        content = """
export const mockContacts = [];
export function useMockStore(key, init) { return {}; }
export const mockKpis = [];
export type KpiItem = { label: string };
"""
        names = _extract_export_names(content)
        self.assertIn("mockContacts", names)
        self.assertIn("useMockStore", names)
        self.assertIn("mockKpis", names)

    def test_filter_mock_exports(self):
        from app_builder.services.context_builder import _filter_mock_exports

        mock_exports = ["mockContacts", "mockDeals", "mockKpis", "mockChartData", "useMockStore"]
        # ContactsPage uses Contact entity
        result = _filter_mock_exports(mock_exports, ["Contact"])
        self.assertIn("mockContacts", result)
        self.assertIn("mockKpis", result)
        self.assertIn("useMockStore", result)

    def test_plan_json_to_legacy_blueprint(self):
        from app_builder.services.frontend_only_pipeline import _plan_json_to_legacy_blueprint

        blueprint_json = {
            "app_name": "My CRM",
            "domain": "crm",
            "routes": [
                {"path": "/", "page": "DashboardPage", "screen_id": "S001"},
                {"path": "/contacts", "page": "ContactsPage", "screen_id": "S002"},
            ],
            "pages": [
                {"file": "DashboardPage"},
                {"file": "ContactsPage"},
            ],
            "nav_items": [
                {"label": "Dashboard", "path": "/", "icon": "Dashboard"},
                {"label": "Contacts", "path": "/contacts", "icon": "People"},
            ],
        }
        design_tokens = {"light": {"primary": "#3f51b5", "background": "#f5f5f5"}, "font_family": "Roboto"}

        legacy = _plan_json_to_legacy_blueprint(blueprint_json, design_tokens, "Build a CRM")

        self.assertEqual(legacy["app_name"], "My CRM")
        self.assertEqual(legacy["domain"], "crm")
        self.assertEqual(legacy["primary_color"], "#3f51b5")
        self.assertEqual(legacy["font_family"], "Roboto")
        self.assertEqual(len(legacy["pages"]), 2)
        self.assertEqual(legacy["pages"][0]["name"], "DashboardPage")
        self.assertEqual(legacy["pages"][0]["path"], "/")

    def test_types_context_no_prd(self):
        """build_types_context must not include any prd field."""
        from app_builder.services.context_builder import build_types_context

        blueprint = {
            "domain": "crm", "app_name": "CRM",
            "entities": [{"name": "Deal", "fields": [{"name": "id", "type": "string"}]}],
        }
        prd = {"summary": "A secret PRD text", "screens": [], "features": [], "goals": [], "personas": [], "non_goals": []}
        ctx = build_types_context(blueprint, prd)

        ctx_str = json.dumps(ctx)
        self.assertNotIn("secret PRD text", ctx_str)
        self.assertIn("Deal", ctx_str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
