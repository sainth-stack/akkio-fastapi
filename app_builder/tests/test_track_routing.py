"""
Unit tests for the frontend-only track routing.

Verifies:
  1. resolve_track() returns "legacy"        for STATIC_DOMAINS
  2. resolve_track() returns "frontend_only" for all other domains
  3. resolve_app_mode() routes "build a CRM app" → "frontend_only" track
  4. resolve_app_mode() routes "online store" / ecommerce → "legacy" track
  5. STATIC_DOMAINS contains exactly the three expected domains
  6. generate_frontend_only_app() returns at least the scaffold files
  7. generate_from_blueprint() returns App.tsx / page stubs / mock/index.ts
"""
from __future__ import annotations

import os
import sys
import unittest

# ---------------------------------------------------------------------------
# Make the fastapi package importable without installing it
# ---------------------------------------------------------------------------
FASTAPI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.abspath(FASTAPI_DIR))


class TestTrackRouting(unittest.TestCase):
    """Tests for STATIC_DOMAINS and resolve_track()."""

    def setUp(self):
        from app_builder.services.fullstack_app_generator import (
            STATIC_DOMAINS,
            resolve_track,
        )
        self.STATIC_DOMAINS = STATIC_DOMAINS
        self.resolve_track = resolve_track

    # ── STATIC_DOMAINS contents ─────────────────────────────────────────────

    def test_static_domains_contains_ecommerce(self):
        self.assertIn("ecommerce", self.STATIC_DOMAINS)

    def test_static_domains_contains_doc_chat(self):
        self.assertIn("doc_chat", self.STATIC_DOMAINS)

    def test_static_domains_contains_anomaly_detection(self):
        self.assertIn("anomaly_detection", self.STATIC_DOMAINS)

    def test_static_domains_has_exactly_three_entries(self):
        self.assertEqual(len(self.STATIC_DOMAINS), 3)

    # ── resolve_track() — legacy ──────────────────────────────────────────

    def test_ecommerce_is_legacy(self):
        self.assertEqual(self.resolve_track("ecommerce"), "legacy")

    def test_doc_chat_is_legacy(self):
        self.assertEqual(self.resolve_track("doc_chat"), "legacy")

    def test_anomaly_detection_is_legacy(self):
        self.assertEqual(self.resolve_track("anomaly_detection"), "legacy")

    # ── resolve_track() — frontend_only ─────────────────────────────────────

    def test_crm_is_frontend_only(self):
        self.assertEqual(self.resolve_track("crm"), "frontend_only")

    def test_analytics_dashboard_is_frontend_only(self):
        self.assertEqual(self.resolve_track("analytics_dashboard"), "frontend_only")

    def test_inventory_is_frontend_only(self):
        self.assertEqual(self.resolve_track("inventory"), "frontend_only")

    def test_finance_is_frontend_only(self):
        self.assertEqual(self.resolve_track("finance"), "frontend_only")

    def test_hr_onboarding_is_frontend_only(self):
        self.assertEqual(self.resolve_track("hr_onboarding"), "frontend_only")

    def test_quality_is_frontend_only(self):
        self.assertEqual(self.resolve_track("quality"), "frontend_only")

    def test_generic_is_frontend_only(self):
        self.assertEqual(self.resolve_track("generic"), "frontend_only")

    def test_unknown_domain_is_frontend_only(self):
        """Any unrecognised domain label falls through to frontend_only."""
        self.assertEqual(self.resolve_track("calculator"), "frontend_only")
        self.assertEqual(self.resolve_track("todo"), "frontend_only")
        self.assertEqual(self.resolve_track(""), "frontend_only")


class TestClassifierRouting(unittest.TestCase):
    """Tests that keyword classification routes to the correct track."""

    def setUp(self):
        from app_builder.services.fullstack_app_generator import (
            _keyword_classify,
            resolve_track,
        )
        self._keyword_classify = _keyword_classify
        self.resolve_track = resolve_track

    def _track_for(self, requirement: str, prd: str = "", uiux: str = "") -> str:
        domain = self._keyword_classify(requirement, prd, uiux)
        return self.resolve_track(domain)

    # "online store" → ecommerce → legacy
    def test_online_store_routes_to_legacy(self):
        track = self._track_for("Build an online store with shopping cart and checkout")
        self.assertEqual(track, "legacy")

    def test_ecommerce_routes_to_legacy(self):
        track = self._track_for("Add to cart, products catalog, order management")
        self.assertEqual(track, "legacy")

    def test_doc_chat_routes_to_legacy(self):
        track = self._track_for("PDF upload and chat Q&A over documents")
        self.assertEqual(track, "legacy")

    def test_anomaly_detection_routes_to_legacy(self):
        track = self._track_for("anomaly detection and outlier alerting on time series metrics")
        self.assertEqual(track, "legacy")

    # "build a CRM" → crm → frontend_only
    def test_crm_routes_to_frontend_only(self):
        track = self._track_for("Build a CRM for managing sales leads and customer contacts")
        self.assertEqual(track, "frontend_only")

    def test_analytics_dashboard_routes_to_frontend_only(self):
        track = self._track_for("analytics dashboard with KPI reporting and business intelligence")
        self.assertEqual(track, "frontend_only")

    def test_calculator_routes_to_frontend_only(self):
        track = self._track_for("Simple calculator app")
        self.assertEqual(track, "frontend_only")

    def test_generic_app_routes_to_frontend_only(self):
        track = self._track_for("A to-do list manager")
        self.assertEqual(track, "frontend_only")


class TestDeterministicGenerator(unittest.TestCase):
    """Tests for generate_from_blueprint()."""

    def setUp(self):
        from app_builder.services.deterministic_generator import generate_from_blueprint
        self.generate_from_blueprint = generate_from_blueprint

    def _sample_blueprint(self):
        return {
            "app_name": "CRM Dashboard",
            "description": "A simple CRM for managing leads and contacts.",
            "pages": [
                {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard", "description": "KPI overview"},
                {"name": "ContactsPage", "path": "/contacts", "nav_label": "Contacts", "icon": "People", "description": "Contact list"},
                {"name": "LeadsPage", "path": "/leads", "nav_label": "Leads", "icon": "TrendingUp", "description": "Sales leads"},
            ],
            "primary_color": "#1976d2",
            "background_color": "#f5f7fb",
            "style": "minimal",
            "domain": "crm",
            "font_family": "Inter",
        }

    def test_returns_dict(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_app_tsx_generated(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        self.assertIn("frontend/src/App.tsx", result)

    def test_app_tsx_has_all_page_imports(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        app_tsx = result["frontend/src/App.tsx"]
        self.assertIn("DashboardPage", app_tsx)
        self.assertIn("ContactsPage", app_tsx)
        self.assertIn("LeadsPage", app_tsx)

    def test_page_stubs_generated(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        self.assertIn("frontend/src/pages/DashboardPage.tsx", result)
        self.assertIn("frontend/src/pages/ContactsPage.tsx", result)
        self.assertIn("frontend/src/pages/LeadsPage.tsx", result)

    def test_mock_index_generated(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        self.assertIn("frontend/src/mock/index.ts", result)

    def test_tokens_generated(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        self.assertIn("frontend/src/theme/tokens.ts", result)
        tokens_ts = result["frontend/src/theme/tokens.ts"]
        self.assertIn("#1976d2", tokens_ts)

    def test_package_json_has_app_name(self):
        result = self.generate_from_blueprint(self._sample_blueprint())
        self.assertIn("frontend/package.json", result)
        pkg = result["frontend/package.json"]
        self.assertIn("crm-dashboard", pkg)

    def test_empty_blueprint_returns_files(self):
        result = self.generate_from_blueprint({})
        self.assertIsInstance(result, dict)
        # No pages → no App.tsx, but tokens + mock should still be written
        self.assertIn("frontend/src/theme/tokens.ts", result)
        self.assertIn("frontend/src/mock/index.ts", result)


class TestFrontendOnlyPipeline(unittest.TestCase):
    """Smoke tests for generate_frontend_only_app().
    Only runs if the base-frontend-vite-mui template is present on disk."""

    def setUp(self):
        import os
        template_dir = os.path.join(
            os.path.dirname(__file__), "..", "templates", "base-frontend-vite-mui"
        )
        self._template_available = os.path.isdir(template_dir)

    def test_template_directory_exists(self):
        self.assertTrue(
            self._template_available,
            "base-frontend-vite-mui template directory not found",
        )

    def test_generate_returns_files(self):
        if not self._template_available:
            self.skipTest("template not present")
        from app_builder.services.frontend_only_pipeline import generate_frontend_only_app
        result = generate_frontend_only_app(
            requirement="Build a simple CRM dashboard",
            prd="",
            uiux="",
            architecture=None,
            design_tokens=None,
            domain="crm",
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_generate_contains_main_tsx(self):
        if not self._template_available:
            self.skipTest("template not present")
        from app_builder.services.frontend_only_pipeline import generate_frontend_only_app
        result = generate_frontend_only_app(
            requirement="Build a simple CRM dashboard",
            domain="crm",
        )
        self.assertIn("frontend/src/main.tsx", result)

    def test_generate_contains_no_backend_files(self):
        if not self._template_available:
            self.skipTest("template not present")
        from app_builder.services.frontend_only_pipeline import generate_frontend_only_app
        result = generate_frontend_only_app(
            requirement="Build a finance tracker",
            domain="finance",
        )
        backend_files = [k for k in result if k.startswith("backend/")]
        self.assertEqual(
            backend_files,
            [],
            f"Expected no backend/ files, found: {backend_files}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
