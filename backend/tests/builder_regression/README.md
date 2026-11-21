# Builder Regression Suite

## What it is

18-prompt regression suite for the Akkio app builder.

- **3 legacy prompts** (ecommerce, doc_chat, anomaly_detection) — must always produce a working fullstack app
- **15 frontend_only prompts** (CRM, HR, finance, inventory, …) — must pass Playwright gate with zero console errors

## Running locally

```bash
# Set env vars
export AKKIO_API_URL=http://localhost:8000
export AKKIO_TEST_TOKEN=<your JWT>

# All prompts
python tests/builder_regression/run_regression.py

# New-track only
python tests/builder_regression/run_regression.py --only new

# Specific prompts
python tests/builder_regression/run_regression.py --ids fo_crm_saas,fo_hr_platform
```

## CI exit codes

| Code | Meaning |
|------|---------|
| 0    | All legacy pass AND new-track ≥ 95 % |
| 1    | Any legacy regressed OR new-track < 95 % |

## Output

Results are written to `tests/builder_regression/results/<timestamp>.json`.

## Pass criteria (per prompt)

| Check | Legacy | New-track |
|-------|--------|-----------|
| Files generated ≥ min_files | ✅ | ✅ |
| tsc --noEmit clean | ✅ | ✅ |
| Playwright gate (all routes open, no blank body, no console errors) | — | ✅ |
| Expected routes in blueprint | — | ✅ |
