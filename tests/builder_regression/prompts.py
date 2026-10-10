"""
Regression suite for the Akkio app builder.

Structure:
  - LEGACY_PROMPTS: 3 prompts that must always produce a working legacy (fullstack) app.
  - NEW_TRACK_PROMPTS: 15 prompts for the frontend_only track.

Each entry is a dict:
  prompt      – user requirement text
  track       – "legacy" | "frontend_only"
  domain      – hint used to verify STATIC_DOMAINS routing
  min_files   – minimum number of generated files expected
  expect_routes – if set, these routes must appear in the blueprint

CI script (run_regression.py) runs every prompt through:
  1. Planning agents  (prd → uiux → design → blueprint)
  2. Code generation pipeline
  3. tsc verify
  4. (frontend_only) Playwright gate
  5. Report pass/fail, build time, tsc errors, console errors
"""

from typing import List, Dict, Any

LEGACY_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "legacy_ecommerce",
        "prompt": "Build an e-commerce platform with product listings, cart, and checkout.",
        "track": "legacy",
        "domain": "ecommerce",
        "min_files": 10,
        "expect_routes": None,
    },
    {
        "id": "legacy_doc_chat",
        "prompt": "Build a document Q&A chatbot where users can upload PDFs and ask questions.",
        "track": "legacy",
        "domain": "doc_chat",
        "min_files": 10,
        "expect_routes": None,
    },
    {
        "id": "legacy_anomaly",
        "prompt": "Build an anomaly detection dashboard for time-series sensor data.",
        "track": "legacy",
        "domain": "anomaly_detection",
        "min_files": 10,
        "expect_routes": None,
    },
]

NEW_TRACK_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "fo_crm_saas",
        "prompt": "Build a CRM SaaS app with leads, deals pipeline (kanban), contacts, and activity feed.",
        "track": "frontend_only",
        "domain": "crm",
        "min_files": 8,
        "expect_routes": ["/leads", "/deals", "/contacts"],
    },
    {
        "id": "fo_project_mgmt",
        "prompt": "Build a project management tool with projects list, task board (kanban), and team members.",
        "track": "frontend_only",
        "domain": "project_management",
        "min_files": 8,
        "expect_routes": ["/projects", "/board"],
    },
    {
        "id": "fo_hr_platform",
        "prompt": "Build an HR platform with employee directory, time-off requests, and payroll summary.",
        "track": "frontend_only",
        "domain": "hr",
        "min_files": 8,
        "expect_routes": ["/employees", "/time-off"],
    },
    {
        "id": "fo_finance_dashboard",
        "prompt": "Build a personal finance dashboard with accounts overview, transactions, budgets, and charts.",
        "track": "frontend_only",
        "domain": "finance",
        "min_files": 8,
        "expect_routes": ["/accounts", "/transactions", "/budgets"],
    },
    {
        "id": "fo_inventory",
        "prompt": "Build an inventory management system with product catalog, stock levels, and low-stock alerts.",
        "track": "frontend_only",
        "domain": "inventory",
        "min_files": 8,
        "expect_routes": ["/products", "/inventory"],
    },
    {
        "id": "fo_analytics",
        "prompt": "Build a marketing analytics dashboard with campaign performance, funnel charts, and KPI cards.",
        "track": "frontend_only",
        "domain": "analytics",
        "min_files": 8,
        "expect_routes": ["/campaigns", "/analytics"],
    },
    {
        "id": "fo_support_tickets",
        "prompt": "Build a customer support ticket system with inbox, ticket detail view, and SLA tracking.",
        "track": "frontend_only",
        "domain": "support",
        "min_files": 8,
        "expect_routes": ["/tickets"],
    },
    {
        "id": "fo_fleet_mgmt",
        "prompt": "Build a fleet management app tracking vehicles, drivers, routes, and maintenance schedules.",
        "track": "frontend_only",
        "domain": "fleet",
        "min_files": 8,
        "expect_routes": ["/vehicles", "/drivers"],
    },
    {
        "id": "fo_healthcare",
        "prompt": "Build a healthcare portal with patient records, appointment scheduling, and lab results.",
        "track": "frontend_only",
        "domain": "healthcare",
        "min_files": 8,
        "expect_routes": ["/patients", "/appointments"],
    },
    {
        "id": "fo_real_estate",
        "prompt": "Build a real estate platform with property listings, agent profiles, and inquiry management.",
        "track": "frontend_only",
        "domain": "real_estate",
        "min_files": 8,
        "expect_routes": ["/properties", "/agents"],
    },
    {
        "id": "fo_subscription",
        "prompt": "Build a subscription management SaaS with plans, billing history, and usage metrics.",
        "track": "frontend_only",
        "domain": "subscription",
        "min_files": 8,
        "expect_routes": ["/plans", "/billing"],
    },
    {
        "id": "fo_learning_mgmt",
        "prompt": "Build a learning management system with course catalog, student progress, and quiz results.",
        "track": "frontend_only",
        "domain": "lms",
        "min_files": 8,
        "expect_routes": ["/courses", "/students"],
    },
    {
        "id": "fo_social_dashboard",
        "prompt": "Build a social media management dashboard with post composer, scheduled posts, and engagement analytics.",
        "track": "frontend_only",
        "domain": "social_media",
        "min_files": 8,
        "expect_routes": ["/posts", "/analytics"],
    },
    {
        "id": "fo_restaurant",
        "prompt": "Build a restaurant management system with table reservations, menu management, and order tracking.",
        "track": "frontend_only",
        "domain": "restaurant",
        "min_files": 8,
        "expect_routes": ["/reservations", "/menu", "/orders"],
    },
    {
        "id": "fo_event_mgmt",
        "prompt": "Build an event management platform with event listings, attendee registration, and check-in.",
        "track": "frontend_only",
        "domain": "events",
        "min_files": 8,
        "expect_routes": ["/events", "/attendees"],
    },
]

ALL_PROMPTS: List[Dict[str, Any]] = LEGACY_PROMPTS + NEW_TRACK_PROMPTS
