# Deploying GraphDigitizer to a New Server

This guide takes a **fresh Linux server** (Ubuntu 22.04/24.04 or Debian 12
assumed) to a running GraphDigitizer instance. The app ships as a **single
container**: one process serves both the FastAPI backend and the built React
frontend on the same origin — no separate frontend host, no nginx required to
get running, no CORS to configure.

> **Sizing.** The container loads PaddleOCR + PaddlePaddle + OpenCV into memory
> and downloads ~100 MB of model weights on first start. Provision at least
> **2 vCPU / 4 GB RAM** and ~5 GB free disk. No GPU is needed (CPU inference).
>
> **Scaling constraint — one worker only.** Session state lives in-process
> (decoded image + detection results). Do **not** add `--workers` or run
> multiple replicas behind a plain load balancer; a `/calibrate` request could
> land on a worker that never saw the matching `/digitize` and 404. Scale
> vertically, or move session state to Redis/disk first.

---

## 1. Prerequisites — install Docker on the server

SSH into the server, then install Docker Engine + the Compose plugin:

```bash
# Docker's convenience script (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sudo sh

# Optional: run docker without sudo (log out/in afterwards to apply)
sudo usermod -aG docker "$USER"

# Verify
docker --version
docker compose version
```

---

## 2. Get the code onto the server

```bash
git clone https://github.com/your-org/graph-digitizer-tool.git
cd graph-digitizer-tool
```

No `.env`, no manual build, no `npm install` on the host — the Docker image
builds the frontend and installs all Python/CV dependencies internally.

---

## 3. Build and start

```bash
docker compose up -d --build
```

This will:
- build the Vite frontend bundle (Node stage) → static files,
- install the Python + CV dependencies and copy the bundle into the runtime image,
- start one single-worker `uvicorn` serving API **and** UI on port 8000,
- create the `paddle-cache` named volume so the ~100 MB OCR weights download
  only once and survive restarts.

First build takes several minutes (dependency install + frontend build). First
**start** also downloads the OCR weights, so the app may take up to a minute to
answer after the container reports up.

---

## 4. Verify it's healthy

```bash
# Container + health status
docker compose ps

# Liveness endpoint (should return JSON, HTTP 200)
curl http://localhost:8000/api/healthz

# Follow startup logs (watch for the PaddleOCR model download to finish)
docker compose logs -f
```

Then open **http://SERVER_IP:8000** in a browser and upload a chart.

---

## 5. Put it behind a domain + HTTPS (recommended for production)

Exposing port 8000 directly is fine for an internal tool. For a public
deployment, terminate TLS at a reverse proxy and keep the app bound to the
Docker network. **Caddy** is the least-effort option (automatic Let's Encrypt
certificates):

Create `/etc/caddy/Caddyfile`:

```
digitizer.example.com {
    reverse_proxy localhost:8000
}
```

```bash
sudo apt install -y caddy        # or: https://caddyserver.com/docs/install
sudo systemctl reload caddy
```

Point the domain's DNS `A` record at the server, open ports 80/443 in the
firewall, and Caddy handles certificates automatically. Because the proxy
forwards to the same app on the same paths, **no `ALLOWED_ORIGINS` change is
needed** — it's still one origin from the browser's perspective.

> If you instead host the frontend on a *different* origin than the API, set
> `ALLOWED_ORIGINS` (see §7) to that frontend's URL so CORS permits it.

### Firewall (optional but sensible)

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80,443/tcp
sudo ufw enable
# With Caddy in front, do NOT expose 8000 publicly.
```

To keep 8000 off the public interface, bind it to localhost only — edit
`docker-compose.yml`:

```yaml
    ports:
      - "127.0.0.1:8000:8000"
```

then `docker compose up -d` to apply.

---

## 6. Day-2 operations

| Task | Command |
|---|---|
| View status / health | `docker compose ps` |
| Tail logs | `docker compose logs -f` |
| Restart | `docker compose restart` |
| Stop | `docker compose down` |
| Stop **and** wipe OCR cache | `docker compose down -v` |
| Update to latest code | `git pull && docker compose up -d --build` |
| Disk usage cleanup | `docker image prune -f` |

The `restart: unless-stopped` policy in `docker-compose.yml` means the container
comes back automatically after a crash or server reboot.

---

## 7. Configuration reference

Set these under `environment:` in `docker-compose.yml` (or via an `.env` file).

| Env var | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS origins. **Only needed for a split deploy** where the frontend is served from a different origin than the API. Leave unset for the single-container setup. |
| `FRONTEND_DIST` | `/app/frontend/dist` (set in the image) | Path to the built frontend the API serves. Don't change unless you relocate the bundle. |
| `HOME` | `/home/appuser` | Home dir of the non-root container user; the OCR model cache lives at `$HOME/.paddleocr` (backed by the `paddle-cache` volume). |

Upload/runtime limits (edit in [api/main.py](../api/main.py) if needed):
`MAX_UPLOAD_BYTES` (25 MB), `SESSION_TTL_SECONDS` (30 min), `MAX_SESSIONS` (64).

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Container `unhealthy` for ~1 min after first start | Normal — OCR weights are downloading. Watch `docker compose logs -f`; it becomes healthy once the model loads. Raise `start_period` in the healthcheck for very slow links. |
| `Permission denied` writing `.paddleocr` on startup | The `paddle-cache` volume mountpoint must be owned by `appuser`; the image pre-creates it. If you added a custom bind-mount for the cache, `chown` it to the UID of `appuser` or use the named volume. |
| Out-of-memory / killed during a request | Bump server RAM to ≥4 GB; large images + OCR are memory-heavy. |
| `502` from the reverse proxy | App still starting (weights downloading) or crashed — check `docker compose logs`. |
| Build fails at `npm ci` | Ensure `frontend/package-lock.json` is committed and present. |
| Uploads rejected as "Unsupported type" | Only PNG/JPEG are accepted; check the `Content-Type`. |

---

## 9. Deploying without Docker (bare host)

If you cannot use Docker, run the two build/run steps directly. Requires
Python 3.10–3.12 and Node 18+.

```bash
# Build the frontend to static files
cd frontend && npm ci && npm run build && cd ..   # → frontend/dist/

# Install backend deps and run (single worker!)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000    # serves frontend/dist automatically
```

For a long-running service, wrap that `uvicorn` command in a systemd unit (with
`Restart=always` and `WorkingDirectory` set to the repo root) and front it with
Caddy/nginx as in §5.
