# SrintellX WhatsApp AI Sales Agent

A production-ready WhatsApp AI sales & support agent for **SrintellX** — the AI
reception system for clinics. It talks to clinic owners, doctors, dentists and
physiotherapists over WhatsApp, explains SrintellX, handles objections, captures
leads, discusses ROI, and books live demos into Google Calendar.

Built with **FastAPI + Python 3.12 + PostgreSQL + OpenAI Responses API**,
deployable on **Railway**.

> The agent answers **only** from the `knowledge_base/` files. It never invents
> pricing, features, or savings figures.

---

## 1. Architecture overview

```
WhatsApp user
     │  (inbound message)
     ▼
Meta WhatsApp Cloud API ──► POST /webhook ──► signature check ──► parse
     ▲                                                              │
     │ (reply)                                                      ▼
     │                                              rate limit · idempotency
     │                                                              │
     │                                                              ▼
WhatsApp client ◄── reply text ◄── AI agent loop ◄── conversation history
                                        │                 (PostgreSQL)
                                        │ tools
              ┌─────────────────────────┼───────────────────────────┐
              ▼            ▼             ▼              ▼             ▼
        capture_lead  log_objection  get_demo_slots  book_demo  escalate
              │            │             │              │
              ▼            ▼             ▼              ▼
           contacts    objections    Google Calendar  demo_bookings
```

The flow:
1. **`webhook.py`** verifies Meta's signature, parses messages, applies a
   per-sender rate limit, and de-duplicates redelivered webhooks.
2. **`conversation_service.py`** upserts the contact and stores the message.
3. Heuristic backstops classify objections and interest (`lead_service.py`).
4. **`ai_service.py`** runs the OpenAI Responses API agent loop with five tools.
   Tool calls update the database and Google Calendar.
5. The reply is persisted and sent back via **`whatsapp.py`**.

### Project structure
```
srintellx-whatsapp-agent/
├── app/
│   ├── main.py                 # FastAPI app, health + admin endpoints
│   ├── config.py               # env-based settings
│   ├── database.py             # async SQLAlchemy engine/session
│   ├── models.py               # contacts, conversations, objections, demo_bookings
│   ├── schemas.py              # WhatsApp webhook + DTO schemas
│   ├── webhook.py              # GET/POST /webhook + processing pipeline
│   ├── whatsapp.py             # Cloud API client + signature verification
│   ├── ai_service.py           # OpenAI Responses agent, tools, KB loader
│   ├── lead_service.py         # lead updates, interest + objection classification
│   ├── calendar_service.py     # Google Calendar slots + booking
│   ├── conversation_service.py # message persistence + history
│   └── utils.py                # logging, phone normalisation, helpers
├── knowledge_base/             # company, pricing, features, faq, objections, roi, demo
├── alembic/                    # migrations (0001_initial)
├── tests/                      # unit tests + example payloads
├── requirements.txt
├── Dockerfile
├── railway.json
└── README.md
```

### Database schema
- **contacts** — one row per WhatsApp sender; progressively filled lead fields,
  `interest_level` (cold/warm/hot), `demo_requested`.
- **conversations** — every inbound/outbound message; `wa_message_id` is unique
  for idempotency.
- **objections** — classified objections with the user's excerpt; de-duplicated
  while unresolved.
- **demo_bookings** — confirmed slots with the Google event id; a unique
  constraint on `(start_time, status)` blocks double-booking at the DB level.

---

## 2. Local development

```bash
# 1. Clone & enter
cd srintellx-whatsapp-agent

# 2. Virtualenv
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env        # then fill in the values

# 4. Start PostgreSQL (example with Docker)
docker run -d --name srintellx-pg -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=srintellx -p 5432:5432 postgres:16

# 5. Run migrations
alembic upgrade head

# 6. Run the API
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the OpenAPI UI.

### Exposing the webhook locally
Use a tunnel (e.g. `ngrok http 8000`) and register the HTTPS URL in the Meta
app dashboard under **WhatsApp → Configuration → Webhook**, with the verify
token from `WHATSAPP_VERIFY_TOKEN` and the `messages` field subscribed.

---

## 3. Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | yes | PostgreSQL URL (Railway injects it). Auto-normalised to asyncpg. |
| `OPENAI_API_KEY` | yes | OpenAI key. |
| `OPENAI_MODEL` | no | Default `gpt-4.1`. |
| `WHATSAPP_TOKEN` | yes | Permanent system-user access token. |
| `WHATSAPP_PHONE_NUMBER_ID` | yes | From WhatsApp API setup. |
| `WHATSAPP_VERIFY_TOKEN` | yes | Any string; must match Meta config. |
| `WHATSAPP_APP_SECRET` | yes (prod) | Meta app secret for signature checks. |
| `WHATSAPP_API_VERSION` | no | Default `v21.0`. |
| `GOOGLE_CALENDAR_ID` | no | Default `primary`. |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | no | Full service-account JSON (one line). |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | no | Path alternative to the JSON var. |
| `DEMO_ORGANIZER_EMAIL` | no | Demo host (Rajesh) added as attendee. |
| `ESCALATION_CONTACT_NAME` | no | Default `Rajesh`. |
| `DEMO_DURATION_MINUTES` | no | Default `30`. |
| `RATE_LIMIT_PER_MINUTE` | no | Inbound msgs per sender/min. Default `20`. |
| `ENVIRONMENT` | no | `development` or `production`. |

> If Google Calendar is **not** configured, booking degrades gracefully to a
> simulated event so the rest of the flow still works.

---

## 4. Google Calendar setup
1. In Google Cloud, create a **service account** and enable the Calendar API.
2. Create a JSON key; paste its contents into `GOOGLE_SERVICE_ACCOUNT_JSON`.
3. Share the target calendar with the service account email (Make changes to
   events permission), and set `GOOGLE_CALENDAR_ID` to that calendar's id.

---

## 5. Deploy to Railway
1. Push this repo to GitHub.
2. On Railway: **New Project → Deploy from GitHub repo**.
3. Add the **PostgreSQL** plugin (provides `DATABASE_URL` automatically).
4. Add all other variables from the table above under **Variables**.
5. Railway builds the `Dockerfile`; the start command runs
   `alembic upgrade head` then launches Uvicorn on `$PORT`.
6. Health check is configured at `/health` in `railway.json`.
7. Point the Meta webhook at `https://<your-app>.up.railway.app/webhook`.

---

## 6. Testing

```bash
source .venv/bin/activate
pytest                       # uses in-memory SQLite; no Postgres needed
```

Covered: webhook parsing, signature verification, objection/interest
classification, contact lifecycle, message idempotency, conversation history,
and objection de-duplication. Example webhook payloads live in
`tests/payloads/`.

### Manual webhook test
```bash
# Verification handshake
curl "http://localhost:8000/webhook?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=123"
# -> 123

# Inbound message (signature check is skipped in development)
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  --data @tests/payloads/inbound_text.json
```

---

## 7. Admin / inspection endpoints
- `GET /health` — liveness + DB check.
- `GET /admin/leads` — recent captured leads.
- `GET /admin/bookings` — recent demo bookings.
- `GET /admin/stats` — counts of leads, bookings, objections.

> Protect `/admin/*` behind authentication or network rules before exposing it
> publicly.

---

## 8. Security notes
- Webhook signatures are validated with HMAC-SHA256 against the app secret
  (enforced in production).
- All secrets come from environment variables; nothing is hard-coded.
- Per-sender rate limiting guards against floods (use Redis for multi-instance).
- Webhook payloads are validated with Pydantic; duplicates are ignored via the
  unique `wa_message_id`.
- The agent is grounded in the knowledge base and instructed never to invent
  pricing or features.

---

## 9. Customising the agent
- Edit the markdown in `knowledge_base/` to change what the agent knows. It is
  reloaded on process start.
- Tune persona/response rules in `PERSONA` inside `app/ai_service.py`.
- Adjust demo windows in `_DEMO_WINDOWS` in `app/calendar_service.py`.
