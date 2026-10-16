import unittest


class TestBlockNormalizer(unittest.TestCase):
    def test_compact_llm_blocks_expand(self):
        from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
        from app_builder.services.block_normalizer import normalize_app_definition_blocks

        defn = AppDefinition(
            meta=AppMeta(appName="Transformer Monitor", description="", domain="generic"),
            navigation=[NavItemDef(label="Analytics", path="/analytics", icon="Assessment", pageId="analytics")],
            pages=[
                PageDef(
                    id="analytics",
                    title="Analytics",
                    blocks=[
                        {"type": "dataTable", "dataKey": "analytics_data"},
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                        },
                        {"type": "lineChart", "dataKey": "performance_trends", "xAxisKey": "label"},
                    ],
                ),
            ],
            mock={
                "analytics_data": [{"id": "1", "name": "A", "status": "OK"}],
                "kpis_dashboard": {"oee": "87%", "throughput": "150 u/h"},
                "performance_trends": [{"label": "W1", "value": 70, "target": 85}],
            },
        )
        out = normalize_app_definition_blocks(defn)
        table_block = out.pages[0].blocks[0]
        self.assertTrue(table_block.get("columns"))
        kpi_block = out.pages[0].blocks[1]
        self.assertEqual(len(kpi_block.get("items") or []), 2)
        chart_block = out.pages[0].blocks[2]
        self.assertEqual(len(chart_block.get("series") or []), 2)

    def test_field_0_columns_replaced_with_maintenance_schema(self):
        from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
        from app_builder.services.block_normalizer import repair_broken_data_table_blocks

        defn = AppDefinition(
            meta=AppMeta(appName="Manufacturing Operations", description="", domain="generic"),
            navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
            pages=[
                PageDef(
                    id="dashboard",
                    title="Dashboard",
                    blocks=[
                        {
                            "type": "dataTable",
                            "title": "Dashboard records",
                            "dataKey": "table_dashboard",
                            "columns": [{"field": "field_0", "header": "Field 0"}],
                        }
                    ],
                )
            ],
            mock={"table_dashboard": [{"field_0": "Field 0 1"}]},
        )
        out = repair_broken_data_table_blocks(defn, prd="manufacturing maintenance")
        cols = out.pages[0].blocks[0]["columns"]
        fields = [c["field"] for c in cols]
        self.assertIn("machine", fields)
        self.assertIn("work_order_id", fields)
        self.assertNotIn("field_0", fields)

    def test_llm_table_columns_title_datakey_repaired(self):
        from app_builder.services.block_normalizer import repair_data_table_columns

        block = repair_data_table_columns(
            {
                "type": "dataTable",
                "dataKey": "maintenance_events_table",
                "columns": [
                    {"title": "Machine", "dataKey": "machine", "field": "", "header": ""},
                    {"title": "Root Cause", "dataKey": "root_cause", "field": "", "header": ""},
                ],
            }
        )
        cols = block["columns"]
        self.assertEqual(cols[0]["field"], "machine")
        self.assertEqual(cols[0]["header"], "Machine")
        self.assertEqual(cols[1]["field"], "root_cause")


if __name__ == "__main__":
    unittest.main()
