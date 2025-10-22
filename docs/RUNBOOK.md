Anagnosis Runbook

Overview
- Services: API (`/api/*`) and Web (HTML + static). Health: `/api/healthz`, `/healthz`.
- Persistence: Postgres (users), `artifacts/` (indexes, chunk rows, per-user prefs/secrets, memory files).
- Concurrency: Single worker per service to avoid FAISS/file write contention.

Logging
- Environment: set `LOG_LEVEL=INFO` (default), or `DEBUG` for troubleshooting.
- Where logs go:
  - With Docker: logs stream to container stdout/stderr (`docker compose logs -f api web`).
  - Without Docker: gunicorn/uvicorn write to stdout; consider redirecting to a file and rotating via `logrotate`.

Health Checks
- API: `GET /api/healthz` → `{ "status": "ok" }`
- Web: `GET /healthz` → `{ "status": "ok" }`

Admin Tools
- Requires a dev/admin user (email in `DEV_EMAILS`). Endpoints under `/api/admin/*`:
  - `POST /api/admin/clear_settings` → remove per-user prefs/secrets files (artifacts/users/*) and clear global config secrets.
  - `POST /api/admin/clear_memory` → remove memory files (`artifacts/memory_*.jsonl`).
  - `POST /api/admin/clear_all` → run both of the above.
  - `POST /api/admin/rebuild_index` → rebuild FAISS index from current rows (no data loss).
  - `GET  /api/admin/users` → list users (id, email) for diagnostics.
  - `GET  /api/admin/usage` → per-user usage counters (ask/ingest), joined with emails.

Backups
- Database (Postgres):
  - Set `DATABASE_URL` and run: `pg_dump "$DATABASE_URL" > anagnosis_$(date +%F).sql`
  - Restore: `psql "$DATABASE_URL" < anagnosis_YYYY-MM-DD.sql`
  - For managed Postgres, use your provider’s snapshot/backup features when possible.
- Artifacts (indexes and files):
  - Snapshot: `tar -czf artifacts_$(date +%F).tgz artifacts/`
  - Restore: `tar -xzf artifacts_YYYY-MM-DD.tgz`

Common Recovery Tasks
- Rebuild index if search quality or counts look off: `POST /api/admin/rebuild_index` (as a dev user).
- Clear a test user’s memory/settings:
  - Memory: `POST /api/admin/clear_memory`
  - Settings: `POST /api/admin/clear_settings`
- Reset cookies or CORS issues:
  - Ensure `ALLOW_ORIGINS` is pinned to your domain.
  - Ensure `COOKIE_SECURE=true` when running behind HTTPS.

Security Notes
- Cookies: HttpOnly + SameSite=Lax by default. Set `COOKIE_SECURE=true` in production (HTTPS required).
- CORS: set `ALLOW_ORIGINS=https://yourdomain.com` (comma‑separated for multiple origins).
- Proxy headers: default nginx config adds common security headers.

Run/Operate Without Docker
- Start API (one worker): `gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 api.main:app`
- Start Web (one worker): `gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:7860 web.app:app`
- Reverse proxy: route `/api/*` to port 8000 and everything else to port 7860. Add TLS at the proxy when on the public Internet.

Troubleshooting
- 401s after login: confirm cookie present; if HTTPS, `COOKIE_SECURE=true`. Ensure proxy preserves cookies and `X-Forwarded-*` headers.
- Upload fails with large PDFs: check `client_max_body_size` (nginx) and confirm `pdf2image`/Tesseract system deps.
- Slow ingestion: keep single workers; large PDFs are CPU/IO bound. Watch logs for timeouts (`ASK_TIME_BUDGET_SEC`).
- 429 Too Many Requests on ask/ingest: default in-memory rate limits apply. Tune with `RATE_ASK_PER_MIN` (default 15) and `RATE_INGEST_PER_MIN` (default 6).

Rate Limiting
- Environment variables:
  - `RATE_ASK_PER_MIN` (default 15)
  - `RATE_INGEST_PER_MIN` (default 6)
  - `RATE_DEFAULT_PER_MIN` (fallback for other actions)
- Scope: enforced per-user ID over a sliding 60s window. Single-process only.
