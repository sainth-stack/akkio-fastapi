from app_builder.services.fullstack_anomaly_generator import is_anomaly_detection_domain
from app_builder.services.json_app_routing import should_force_json_app_track

USER_PROMPT = """
Create a manufacturing maintenance dashboard with 3 pages: Dashboard, Analytics, Reports.
Include an in-app data assistant (RAG) over KPIs and charts.
Add a Predict page for synthetic what-if forecasts on MTBF, downtime, and work orders.
No CSV/export buttons on dashboard pages.
""".strip()


def test_maintenance_prompt_forces_json_app():
    assert should_force_json_app_track(USER_PROMPT)


def test_maintenance_not_classified_as_anomaly():
    assert not is_anomaly_detection_domain(USER_PROMPT, prd="metrics and alerts monitoring")
