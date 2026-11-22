import copy
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
from app_builder.schemas.app_definition_llm import AppShellPlan, PageBlocksPlan, PageDataPlan
from app_builder.services.generated_data_validation import GeneratedDataValidationError, validate_generated_definition


class TestGeneratedDataValidation(unittest.TestCase):
    def setUp(self):
        for transport, method in ((httpx.HTTPTransport, "handle_request"), (httpx.AsyncHTTPTransport, "handle_async_request")):
            guard = patch.object(transport, method, side_effect=AssertionError("Live HTTP forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def definition(self):
        return AppDefinition(
            meta=AppMeta(appName="Stock Pulse", domain="inventory", generatedBy="llm_pipeline"),
            navigation=[NavItemDef(label="Inventory", path="/", pageId="inventory")],
            pages=[PageDef(id="inventory", title="Inventory", blocks=[
                {"type": "dataTable", "title": "Warehouse stock", "dataKey": "stock", "columns": [
                    {"field": "sku", "header": "SKU"}, {"field": "quantity_on_hand", "header": "Quantity"},
                ]},
            ])],
            mock={"stock": [{"sku": "SKU-901", "quantity_on_hand": 321}]},
        )

    def test_finalize_and_serialize_preserve_single_table(self):
        from app_builder.services.app_definition_generator import finalize_app_definition
        from app_builder.services.json_app_generator import generate_json_app_files
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        original = self.definition()
        expected = copy.deepcopy(original.mock)
        result = finalize_app_definition(original, requirement="Inventory management with SKUs and warehouses")
        self.assertEqual([b["type"] for b in result.pages[0].blocks], ["dataTable"])
        self.assertEqual(result.mock, expected)
        self.assertEqual(result.meta.dataModel, "llm_generated_v1")
        fields = [f["key"] for f in result.meta.predictConfig["fields"]]
        self.assertIn("quantity_on_hand", fields)
        self.assertNotIn("sku", fields)
        self.assertNotIn("mtbf_hours", fields)
        payload = json.loads(generate_json_app_files(app_definition=result)["frontend/src/mock/data.json"])
        self.assertEqual(payload["data"], expected)
        again = apply_source_data_pipeline(payload)
        self.assertEqual(again["data"], expected)
        self.assertEqual(again["pages"], payload["pages"])

    def test_invalid_table_data_fails_without_synthesis(self):
        from app_builder.services.block_normalizer import align_table_datasets_to_columns

        for data in ({}, {"stock": []}, {"stock": [{"sku": "SKU-901", "quantity": 321}]},
                     {"stock": [{"sku": "Sample sku 1", "quantity_on_hand": 321}]}):
            with self.subTest(data=data):
                definition = self.definition()
                definition.mock = copy.deepcopy(data)
                with self.assertRaises(GeneratedDataValidationError):
                    align_table_datasets_to_columns(definition, llm_authored=True)
                self.assertEqual(definition.mock, data)

    def test_missing_kpi_is_not_fabricated(self):
        from app_builder.services.production_dataset_enhancer import enhance_production_datasets

        definition = self.definition()
        definition.pages[0].blocks = [{"type": "kpiRow", "dataKey": "kpis_inventory", "items": [
            {"label": "Inventory value", "valueKey": "inventory_value"},
        ]}]
        with self.assertRaises(GeneratedDataValidationError):
            enhance_production_datasets(definition)
        self.assertNotIn("kpis_inventory", definition.mock)

    def test_all_chart_series_and_rows_are_validated(self):
        definition = self.definition()
        definition.pages[0].blocks.append({"type": "lineChart", "dataKey": "trend", "xAxisKey": "month", "series": [
            {"dataKey": "stock_level", "name": "Stock level"},
        ]})
        definition.mock["trend"] = [{"month": "Jan", "stock_level": 10}, {"month": "Feb", "stock_level": 12}]
        validate_generated_definition(definition)
        definition.mock["trend"][1]["stock_level"] = "twelve"
        with self.assertRaises(GeneratedDataValidationError):
            validate_generated_definition(definition)

    def test_secondary_llm_path_marks_provenance_before_finalize(self):
        from app_builder.schemas.app_definition_gen import AppDefinitionGen
        from app_builder.services.app_definition_generator import _gen_to_app_definition

        original = self.definition()
        gen = AppDefinitionGen(appName="Stock Pulse", domain="inventory", navigation=[n.model_dump() for n in original.navigation],
                       pages=[p.model_dump() for p in original.pages], mock=original.mock)
        result = _gen_to_app_definition(gen, original.theme)
        self.assertEqual(result.meta.generatedBy, "llm")
        self.assertEqual(result.mock, original.mock)

    def test_frontend_has_no_synthetic_table_repair(self):
        root = Path(__file__).parents[1] / "templates/base-frontend-vite-mui/frontend/src/renderer"
        self.assertNotIn("repairTableRows", (root / "BlockRenderer.tsx").read_text(encoding="utf-8"))
        self.assertNotIn("MAINTENANCE_TABLE_COLUMNS", (root / "blockNormalize.ts").read_text(encoding="utf-8"))
        self.assertFalse((root / "tableDataRepair.ts").exists())


class TestMockedGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_data_repaired_with_mocked_llm(self):
        await self.run_generation(repair=True)

    async def test_exhausted_data_repair_fails(self):
        await self.run_generation(repair=False)

    async def run_generation(self, repair):
        from app_builder.services.app_definition_llm_pipeline import generate_app_definition_llm

        definition = TestGeneratedDataValidation().definition()
        shell = AppShellPlan(appName="Stock Pulse", domain="inventory", navigation=[n.model_dump() for n in definition.navigation],
                             pages=[{"id": "inventory", "title": "Inventory"}])
        blocks = PageBlocksPlan(blocks=definition.pages[0].blocks)
        async def response(llm, system, user, schema, label):
            if schema is AppShellPlan:
                return shell
            if schema is PageBlocksPlan:
                return blocks
            if label.endswith("_0") or not repair:
                return PageDataPlan(data={})
            return PageDataPlan(data=definition.mock)

        mocked = AsyncMock(side_effect=response)
        with patch("llm_helper.get_llm_for_user", return_value=object()), \
             patch("app_builder.services.app_definition_llm_pipeline._llm_validated", mocked), \
               patch("app_builder.services.json_app_codegen_pipeline._tick", new=AsyncMock()), \
             patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("Live HTTP forbidden")), \
             patch.object(httpx.AsyncHTTPTransport, "handle_async_request", side_effect=AssertionError("Live HTTP forbidden")):
            if repair:
                from app_builder.services.json_app_generator import run_json_app_codegen

                files = await run_json_app_codegen(
                    requirement="Inventory management with SKUs and warehouses", plan_json={}, project_name="offline-audit",
                )
                result = json.loads(files["frontend/src/mock/data.json"])
                self.assertEqual(result["data"], definition.mock)
                self.assertEqual(result["pages"][0]["blocks"][0]["type"], "dataTable")
            else:
                with self.assertRaises(GeneratedDataValidationError):
                    await generate_app_definition_llm(requirement="Inventory management with SKUs and warehouses")
        labels = [call.args[-1] for call in mocked.await_args_list if call.args[3] is PageDataPlan]
        self.assertEqual(len(labels), 2 if repair else 3)