# Prompt-driven manufacturing builder

New JSON apps use `generationPolicy: strict_llm_v1`. Their layouts and datasets are validated LLM output. Missing blocks or datasets fail generation instead of invoking deterministic page/data substitutes.

## Runtime

- `runtime_definition.json` stores the current definition separately from bundled frontend assets.
- `/api/apps/{project}/definition` loads persisted state.
- `/api/apps/{project}/data/import` imports declared table datasets and requires project authentication.
- `/api/apps/{project}/data/records/{dataKey}` creates or replaces validated table records and requires project authentication.
- Writes use atomic file replacement and an in-process project lock. Multi-worker deployments still need a transactional shared datastore before enabling concurrent mutations.
- A new successful generation replaces the runtime definition. Export user-entered data before regenerating an existing app.

## Supported generated contracts

Pages may contain tables, KPI rows, line/bar/pie charts, text, select/date/search filters, create-record forms, CSV export and browser print-to-PDF.

KPI and line/bar chart blocks can declare `sourceDataKey` referencing a table. KPI items and chart series declare `aggregation`: `count`, `sum`, `avg`, `min`, `max`, or `ratio`. Numeric operations require `field`; ratio requires `numeratorField`, `denominatorField`, and optional `scale`. Equality predicates use `where`. Charts require `groupBy` referencing a table column.

Only explicit bindings are computed. Unknown fields or formulas fail instead of guessing. Empty numeric inputs produce null metrics. Unbound LLM metrics remain snapshots and do not imply automatic recalculation. Filters apply to matching dataset fields; preaggregated KPI rows are not recalculated by client-side filtering.

Synthetic data is labeled. Actual factory data must be supplied through uploads or a separately implemented integration. Advanced scheduling solvers, approval permissions, arbitrary workflow transitions, MES/ERP connectors and relational transaction rules are not implemented by the JSON renderer; they require additional fullstack contracts and verification.

Legacy static-domain and synchronous scaffold helpers retain their existing behavior. Existing apps must be regenerated to adopt the strict policy. These changes do not establish LLM-only behavior for every legacy/update endpoint.

## Validation

- `python -m pytest app_builder/tests/test_strict_manufacturing_builder.py app_builder/tests/test_json_app_routing.py app_builder/tests/test_json_app_template_guard.py -q`
- In the frontend template: `npm run typecheck` and `npm run build`.
- Live generation still requires configured LLM access, authentication and the backend's full dependencies.

The focused suite covers generation failure, prompt/model propagation, nested PRD sections, source-bound metrics, status/zero preservation, empty data, imports and durable records. It does not replace real manufacturing acceptance testing or multi-user deployment testing.