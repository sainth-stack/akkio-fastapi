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


if __name__ == "__main__":
    unittest.main()
