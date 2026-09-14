# Jarvis

A small personal assistant powered by the [Google AI Studio](https://aistudio.google.com/apikey)
Gemini API. Streams replies token by token, keeps conversation history, and runs
either as a terminal REPL or as a web app.

```
jarvis/
  config.py    settings from the environment / .env
  memory.py    conversation history + JSON persistence
  llm.py       Gemini streaming client (transport is injectable)
  cli.py       terminal chat
  server.py    FastAPI app: SSE chat API, runtime key/model switch, auth gate
  security.py  key redaction + masking
  web/         the chat UI (no build step, no JS dependencies)
tests/         89 tests, all offline — no API key or network needed
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env                                  # then paste your key
```

`.env` holds `GEMINI_API_KEY`. It is git-ignored — never commit it.

> **Key format:** AI Studio now issues **Auth keys** starting with `AQ.` instead of
> the legacy `AIza` "traffic keys". Both work here — Jarvis passes the key straight
> to `google-genai` (which sends it as the `x-goog-api-key` header) and never
> validates its shape. There is a regression test guarding that. Note that `AQ.`
> keys are rejected by OpenAI-compatible endpoints; Jarvis uses the native Gemini
> endpoint, so this does not apply.

## Run

**Web app** — <http://localhost:8000>

```bash
python -m jarvis serve                 # or: python -m jarvis serve --port 8080
```

**Terminal**

```bash
python -m jarvis                       # interactive REPL
python -m jarvis chat -m "summarise this"   # one-shot
```

Commands inside the REPL: `exit` quits, `reset` clears history.

### No key yet?

`JARVIS_MODEL=mock python -m jarvis serve` runs the whole stack offline with a
canned responder, so you can develop the UI without burning quota.

## Configuration

| Variable              | Default               | Meaning                                   |
| --------------------- | --------------------- | ----------------------------------------- |
| `GEMINI_API_KEY`      | —                     | Your Google AI Studio key                  |
| `JARVIS_MODEL`        | `gemini-2.5-flash`    | Any Gemini model, or `mock` for offline    |
| `JARVIS_SYSTEM_PROMPT`| concise-assistant prompt | Persona instructions                   |
| `JARVIS_MAX_TURNS`    | `40`                  | Messages kept in context                   |
| `JARVIS_HOST`         | `0.0.0.0`             | Bind address                               |
| `JARVIS_PORT`         | `8000`                | Port                                       |
| `JARVIS_DATA_DIR`     | `./data`              | Where `history.json` lives                 |
| `JARVIS_AUTH_TOKEN`   | — (auth off)          | When set, every POST needs `Authorization: Bearer <token>` |
| `JARVIS_EXPOSE_DOCS`  | auto                  | Force `/docs` + `/openapi.json` on (`1`) or off (`0`). Auto = off when a token is set |

## API

| Method | Path          | Description                                            |
| ------ | ------------- | ------------------------------------------------------ |
| GET    | `/api/health` | Model, provider, key status, current messages           |
| POST   | `/api/chat`   | `{"message": "…"}` → SSE stream of `start`/`token`/`done`/`error` |
| POST   | `/api/key`    | `{"api_key": "…"}` → held in memory only, never written to disk |
| POST   | `/api/model`  | `{"model": "…"}` → switch model at runtime, e.g. `mock`  |
| POST   | `/api/reset`  | Clear the conversation                                  |

Example:

```bash
curl -N -X POST localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello"}'
```

Everything is configurable from the browser: the Settings panel sets the key,
the model, and the route — no restart needed. Setting the model to `mock` gives
you a working chat loop with zero API calls.

## Two ways to reach Google

The Settings panel in the UI offers both:

* **Through this server** (default) — the browser talks to Jarvis, Jarvis talks to
  Google. The key stays server-side. Needs outbound HTTPS from the server.
* **Browser → Google direct** — the browser calls the Gemini REST API itself, with
  the key kept in `localStorage`. Use this when the machine hosting Jarvis has no
  route to `generativelanguage.googleapis.com` (locked-down networks, sandboxes).

## Tests

```bash
python -m pytest          # 89 passed
```

The Gemini transport is driven by a stubbed client in tests, so the request
payload, chunk parsing, and error mapping are covered without network access.

## Key protection

The API key is write-only. Once loaded it cannot be read back out:

* **Never returned.** No endpoint echoes the key. `/api/health` reports only
  `api_key_configured` plus a masked hint such as `AQ.Ab8…000`.
* **Never on disk.** The configured key lives in process memory; only the
  conversation is persisted. A test asserts the key appears in no file under
  the data dir.
* **Not even if you paste it into the chat box.** `scrub_message()` blanks
  credential-shaped tokens (`AQ.`, `AIza`, `sk-`, `ghp_`) from a message
  *before* it is sent to the model or written to `history.json`, and a message
  that was nothing but a credential is rejected with 422. Without this, a key
  typed into the chat was persisted verbatim — and echoed back by the reply.
* **Never in errors.** `jarvis/security.py` redacts the key — plus `key=`
  query params, `x-goog-api-key` and `Authorization: Bearer` values — from
  every message before it reaches the browser, the terminal, or a log line.
  SDK/HTTP exceptions can quote the request they failed on; this is why.
* **Not spendable by strangers.** Set `JARVIS_AUTH_TOKEN` and `/api/chat`,
  `/api/key`, `/api/model` and `/api/reset` all require the bearer token
  (constant-time compared). `/api/health` and the UI stay open so a page can
  still discover that a token is needed. Without it, anyone who can reach the
  port can use your key.
* **Smaller surface when public.** Setting `JARVIS_AUTH_TOKEN` also hides
  `/docs`, `/redoc` and `/openapi.json`, which otherwise publish the whole API
  shape to anyone. Override with `JARVIS_EXPOSE_DOCS=1`.
* **History is `0600`.** `data/history.json` is created through `tempfile` and
  readable only by its owner.

Audited and found clean: uvicorn access logs record `POST /api/key ... 200 OK`
with no key value, and `OPTIONS /api/key` returns 405 with no CORS headers, so
no other origin can drive the API from a browser.

* **Browser side is opt-in.** A browser-direct key goes to `sessionStorage` and
  dies with the tab; the "Remember the key" checkbox is what moves it to
  `localStorage`.

Verified against a live server: with a key loaded, `GET /api/health`, `GET /`,
`GET /static/app.js` and a streaming `POST /api/chat` all return responses
containing no part of it.

## Durability notes

* Replies are flushed to `data/history.json` as tokens arrive (throttled to
  ~2/s), so a client that hangs up mid-reply keeps the question and the partial
  answer instead of losing the turn.
* A failed request (bad key, quota, network) is rolled back and leaves no
  half-written history behind.
* History writes are atomic (`tempfile` + `replace`), and a corrupt history file
  is ignored rather than crashing startup.
