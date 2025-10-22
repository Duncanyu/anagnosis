Small-Scale Deployment (single instance)

Overview
- One API process (FastAPI, port 8000)
- One Web process (FastAPI, port 7860)
- Fronted by a reverse proxy on 80/443 routing:
  - `/api/*` → API
  - everything else → Web
- Postgres for DB
- Persistent volume for `artifacts/`

Prereqs
- A Linux host with Docker + Docker Compose v2
- A domain name pointing to your host (A/AAAA records)
- Optional provider keys if you’ll use OpenAI / SerpAPI / Brave

Environment
1) Copy `.env.example` to `.env` and set:
   - `SECRET_KEY` → strong random string
   - `ALLOW_ORIGINS` → `https://yourdomain.com`
   - `COOKIE_SECURE` → `true` (required for HTTPS)
   - `DEV_EMAILS` → comma‑separated admin emails
   - `DATABASE_URL` → leave compose default for local Postgres, or set managed Postgres URL

2) Ensure `artifacts/` directory exists (compose will mount it):
```
mkdir -p artifacts
```

Option A — TLS via Caddy (automatic Let’s Encrypt)
1) Edit `docker/Caddyfile` and set your domain:
   - Replace `{$DOMAIN}` with your FQDN (or export `DOMAIN` env before compose)
   - Optionally set contact email at the top

2) Use the provided override file to run Caddy instead of nginx:
```
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
```

3) Visit `https://yourdomain.com` → login/signup should work; `/api/healthz` and `/healthz` return OK.

Option B — TLS via Nginx + external certs
1) Use the default `docker-compose.yml` (nginx on port 8080) for local testing:
```
docker compose up --build
open http://localhost:8080
```

2) For production TLS, adapt `docker/nginx.conf` to include your certs and listen on 443, or terminate TLS at a cloud load balancer that forwards to nginx on 80.

Postgres (managed)
- If you prefer a managed DB, set `DATABASE_URL` in `.env` to your provider URL and remove the `postgres` service (and references) from `docker-compose.yml`.

Operational Notes
- Single worker (`-w 1`) is intentional to avoid FAISS/file contention; adequate for 2–3 concurrent users.
- Artifacts are persisted on the host via `./artifacts`. Back this up if needed.
- Health checks: `/api/healthz` and `/healthz` return `{ "status": "ok" }`.
- Admin tools: Dev users (in `DEV_EMAILS`) see a Dev tab to clear per‑user settings and memory.
 - Security headers: The default nginx adds `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and `Referrer-Policy: no-referrer`. If you terminate TLS in nginx, you can enable HSTS (see comment in `docker/nginx.conf`).

Common Env Vars
- `SECRET_KEY` — JWT signing secret
- `ALLOW_ORIGINS` — CORS allow‑list (comma‑separated)
- `COOKIE_SECURE` — `true` to mark cookies as Secure (HTTPS only)
- `DEV_EMAILS` — list of dev/admin emails
- `DATABASE_URL` — SQLAlchemy Postgres URL (compose default points to local container)
- Provider keys (optional): `OPENAI_API_KEY`, `SERPAPI_KEY`, `BRAVE_API_KEY` (users can also set per-user keys in Settings)

Security Checklist (Phase 5)
- HTTPS enabled (Caddy or nginx+certs) and valid certificate
- `COOKIE_SECURE=true` in production so auth cookies are Secure + HttpOnly + SameSite=Lax
- `ALLOW_ORIGINS` pinned to your exact domain (e.g., `https://yourdomain.com`)
- Confirm browser shows security headers and cookies set as expected

Rollout Checklist
1) DNS points to host
2) `.env` configured
3) Compose up (with TLS option of your choice)
4) Create 2 test users; verify isolated libraries + ask results
5) Confirm cookies are `Secure` and CORS is limited to your domain
