"""E-commerce vertical detection (electronics vs pharmacy)."""
from __future__ import annotations

import os
import sys
import unittest

FASTAPI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.abspath(FASTAPI_DIR))


class TestEcommerceVertical(unittest.TestCase):
    def test_electronics_wins_over_pharma_prd(self):
        from app_builder.services.fullstack_ecommerce_generator import _detect_product_domain

        req = "ecommerrce electronicd app"
        prd = "Online pharmacy with healthcare, vitamins, and wellness products."
        self.assertEqual(_detect_product_domain(req, prd, ""), "electronics")

    def test_pharma_catalog_when_prompt_says_pharmacy(self):
        from app_builder.services.fullstack_app_generator import generate_fullstack_application

        files = generate_fullstack_application(
            "Build an online pharmacy store with shopping cart, checkout, and product listing",
            "Medicine and wellness delivery for customers",
            "",
            {},
            forced_mode="ecommerce",
        )
        mock = files.get("frontend/src/api/mock.ts", "")
        self.assertIn("Paracetamol", mock)
        self.assertNotIn("Laptop", mock)

    def test_legacy_forced_mode_ignores_llm_generic(self):
        from app_builder.services.fullstack_app_generator import generate_fullstack_application

        files = generate_fullstack_application(
            "online store with cart and checkout",
            "",
            "",
            {},
            forced_mode="ecommerce",
        )
        self.assertIn("frontend/src/api/mock.ts", files)
        self.assertNotIn("frontend/src/mock/data.json", files)

    def test_electronics_products_in_generated_mock(self):
        from app_builder.services.fullstack_app_generator import (
            generate_fullstack_application,
            resolve_app_mode,
        )

        req = "Build an ecommerce store for electronic gadgets and laptops"
        self.assertEqual(resolve_app_mode(req, "", ""), "ecommerce")
        files = generate_fullstack_application(
            req,
            "Pharmacy and medicine delivery with healthcare supplements",
            "",
            {},
        )
        mock = files.get("frontend/src/api/mock.ts", "")
        self.assertIn("Laptop", mock)
        self.assertNotIn("Paracetamol", mock)


if __name__ == "__main__":
    unittest.main(verbosity=2)
