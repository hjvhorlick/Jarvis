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
  server.py    FastAPI app: SSE chat API, runtime key/model switch
  web/         the chat UI (no build step, no JS dependencies)
tests/         53 tests, all offline — no API key or network needed
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env                                  # then paste your key
```

`.env` holds `GEMINI_API_KEY`. It is git-ignored — never commit it.

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
python -m pytest          # 53 passed
```

The Gemini transport is driven by a stubbed client in tests, so the request
payload, chunk parsing, and error mapping are covered without network access.

## Durability notes

* Replies are flushed to `data/history.json` as tokens arrive (throttled to
  ~2/s), so a client that hangs up mid-reply keeps the question and the partial
  answer instead of losing the turn.
* A failed request (bad key, quota, network) is rolled back and leaves no
  half-written history behind.
* History writes are atomic (`tempfile` + `replace`), and a corrupt history file
  is ignored rather than crashing startup.
