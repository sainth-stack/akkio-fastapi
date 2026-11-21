# Prism — Gemini image-generation POC

Standalone React/TypeScript/Vite frontend and Node/TypeScript/Express backend. Existing Akkio FastAPI code is unchanged. Production requests always use real Gemini browser automation; intercepted fixtures exist only under `tests/`.

## Architecture

React → POST job with UUID idempotency key → persisted job manager → dedicated persistent Chromium → Gemini authentication check → single prompt submission → newest model response → original browser download → signature/MIME consistency and Sharp decode validation → atomic filesystem storage → persisted COMPLETED → polling API → actual image preview.

Only one generation runs at a time. Other requests get HTTP 409. Duplicate idempotency keys return the same job. The UI persists both pending submission keys and accepted job IDs for refresh recovery. Metadata includes dimensions, byte length, MIME, SHA-256, timestamps, and provider. Restarted incomplete jobs become FAILED rather than being resubmitted.

```text
poc/
  backend/src/
    app.ts, server.ts, config.ts, auth.ts, errors.ts
    automation/       browserManager.ts, geminiPage.ts
    jobs/             generationJobManager.ts
    services/         geminiAutomation, imageDownload, imageStorage
  frontend/src/
    components/       PromptInput, GenerationStatus, GeneratedImage, GenerationHistory
    pages/            ImageGenerator.tsx
    services/         imageGeneration.api.ts
  shared/             types.ts
  tests/              unit/, integration/, fixtures/, e2e/
  .local/             private profiles, jobs, images, diagnostics (gitignored)
```

## Installation (PowerShell)

Use Node 22 LTS (22.22.2+) or Node 24 LTS (24.15.0+). Node 23 on this machine executed the tests but is not an officially supported Vitest runtime. Commands are run from the POC directory.

```powershell
Set-Location C:\Users\SG0709464\dx\akkio\akkio-fastapi\poc
npm ci
npm run browser:install
Copy-Item .env.example .env
npm run auth
```

The sign-in command opens a dedicated Chromium window. Enter all Google credentials and security checks only in that window. Once Gemini is open, press Enter in the terminal to verify the session and close Chromium. The profile retains session cookies, not a password supplied to this app. Do not copy an unrelated Chrome profile or export credentials. Close the sign-in command before generating images.

Local npm 10.9.9 encountered an `edgesOut` installer crash during the Vitest upgrade. Using a newer npm resolved it; use a supported Node LTS distribution with current npm if this recurs.

## Development and production build

The app now includes **Sign in to Gemini**. Click it to open headed Chromium using the same profile as image generation. Click Google’s Sign in in that window and enter your email/password only on Google’s page. Complete any checks, return to Prism, and click **I have signed in**. Successful verification saves the session and closes the browser; failed verification leaves it open. **Cancel sign-in** closes it without reporting authentication. Refreshing Prism retains the current sign-in state through backend polling. Then click Generate or Retry generation. This is a local desktop workflow: the browser opens on the backend machine, not on a remote client.

`GET /api/gemini-session` returns the sign-in state. `POST /api/gemini-session/start`, `/confirm`, and `/cancel` control the manual browser lifecycle. These endpoints accept no credentials. Image generation is blocked while sign-in is open.

```powershell
npm run dev
```

Frontend: http://127.0.0.1:5180/ . Backend: http://127.0.0.1:5181/api/health . Vite proxies `/api` to the backend; use `127.0.0.1`, not `localhost`, to match the configured origin. Separate startup commands are `npm run dev:backend` and `npm run dev:frontend`.

```powershell
npm run build
npm start
```

The built frontend is also served at http://127.0.0.1:5181/ . Keep `HEADLESS=false` initially so security checks and UI changes can be diagnosed. Restart the backend after changing `.env`. Vite’s development proxy is fixed at port 5181; update `vite.config.ts` if changing the backend port.

## Tests

Stop the development servers before running fixture E2E tests: the harness refuses to reuse an existing server so it cannot accidentally test a production provider.

```powershell
npm test
npm run test:e2e
npm run build
npm audit
```

`npm test`: image validation/storage, React states, API integration, persistence, idempotency, serialization, and Chromium automation against intercepted Gemini pages. All fixture browser requests are intercepted; they cannot contact Google. Synthetic fixture images are test data, not real Gemini output.

Real integration requires the authenticated dedicated profile and running `npm run dev` in a separate terminal:

```powershell
$env:LIVE_GEMINI='1'
npm run test:live
Remove-Item Env:LIVE_GEMINI
```

The live test submits through the real UI/API, requires COMPLETED, downloads the backend image, validates it, attaches actual image bytes to the test report, and captures `test-results/live-studio.png`. It never switches to fixtures. Without `LIVE_GEMINI=1`, it is skipped, not passed. Live tests consume actual Gemini account usage.

## API

- `POST /api/image-generation`: strict JSON `{ "prompt": "..." }`, trimmed 1–4000 characters by default; optional UUID `Idempotency-Key` header; HTTP 202 `{jobId,status}`.
- `GET /api/image-generation/:jobId`: persisted public job metadata; 404 if missing.
- `GET /api/image-generation`: newest 100 jobs for local history.
- `GET /api/generated-images/:filename`: only registered completed job images; UUID allowlist and no path traversal; `?download=1` sets attachment disposition.
- `GET /api/health`: process availability, busy state, provider; it does not claim the Gemini session is authenticated.

400 invalid prompt/JSON/key, 403 foreign host/origin, 409 active job/key conflict, 413 oversized body, 415 unsupported content type, 429 rate limit, 503 job persistence/profile-related submission failure. Automation failures are returned as FAILED jobs with an understandable error code/message.

## Download guarantees and storage

The service identifies an image only in a newly generated model response, waits for generation to stop, and uses the response’s download button or image dialog’s download button. It does not save a screenshot, fetch an arbitrary URL, or substitute the displayed thumbnail. A changed download UI produces failure.

Browser Download does not expose response headers. The MIME assertion is derived from the original suggested filename, then checked against PNG/JPEG/WebP signatures and full image decoding. Unsupported formats, HTML pages, truncated files, files over 25 MiB, and images below 256 × 256 are rejected. Original bytes are never transcoded. Sharp pixel limits restrict decompression costs. Storage checks the saved bytes before success. An `ImageStorage` implementation can replace `LocalImageStorage` for S3/GCS; no object storage adapter is included.

## Security and limitations

- Local-only service, not a multi-user deployment. Loopback binding, Host/Origin checks, rate limiting, strict body validation, UUID filenames, Helmet, no public profile/diagnostic routes, no credentials in code.
- Prompts/history are private local JSON; screenshots may contain account information. Restrict OS access to `.local`; never commit/share its contents. Windows ACLs, not POSIX `mode`, control effective filesystem permissions on Windows.
- Navigation retry occurs only before submission. An uncertain submission is never automatically retried. Download failures are not blindly retried; check Gemini before clicking Retry.
- Gemini UI selectors assume English and current response semantics (`model-response` or explicitly marked model response elements). Account or UI changes can cause fail-closed behavior; authenticated response/download selectors remain unverified against a real signed-in session until the live success test passes.
- Model availability, subscription requirements, regional availability, content policy, CAPTCHA, and account limits are controlled by Google. Respect Google’s terms and restrictions. No bypasses or stealth patches are implemented.
- Jobs are retained on disk; no pruning, cancellation endpoint, queue, multiple users, distributed locking, or production authentication is included. A process crash may leave `.poc-lock`; remove it only after confirming no POC/auth browser process owns the profile. Do not run multiple POC servers against the same job/storage directories.
- External Google Fonts are cosmetic only; browser tests block them and use fallback fonts. Clipboard copy requires browser permission/secure context.

## Troubleshooting

`AUTH_REQUIRED` / `AUTH_UNVERIFIED`: click **Sign in to Gemini** (or run `npm run auth`), complete sign-in in **Google Chrome**, then verify. If account verification fails after successful login, inspect the private screenshot and update accessible account selectors; do not remove authentication checks.

`INSECURE_BROWSER` / “This browser or app may not be secure”: Google rejected Playwright Chromium. This POC launches installed Google Chrome with a fresh `.local/chrome-profile`. Close the blocked window, **Quit Chrome (Cmd+Q)** if needed, then sign in again. Do not use Chromium-for-Testing for Google login.

`PROFILE_BUSY`: close auth/other POC browsers. After a crash, confirm processes are stopped before deleting `.local/gemini-profile/.poc-lock`.

`BROWSER_LAUNCH_FAILED`: install Chromium; verify profile permissions and available desktop display when headed.

`GEMINI_UI_UNAVAILABLE` / `GEMINI_UNAVAILABLE`: check connectivity, Google availability, and `.local/diagnostics/<jobId>.png`.

`PROMPT_NOT_FOUND` / `DOWNLOAD_CONTROL_NOT_FOUND`: inspect the real UI and update role/label selectors. Never silently replace original downloads with thumbnails.

`GENERATION_TIMEOUT` / `SUBMISSION_UNCERTAIN`: inspect the Gemini conversation before retrying; the request may already have consumed usage.

`INVALID_IMAGE` / `DOWNLOAD_FAILED`: no COMPLETED status is published. Inspect the browser download UI and private failure screenshot.

`STORAGE_FAILED`: check free space and write permissions for configured storage paths. API responses never include these paths.

See `TEST_REPORT.md` for the actual results from this execution.