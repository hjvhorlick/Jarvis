/**
 * Drives the real jarvis/web/app.js inside a jsdom DOM and asserts what it
 * actually sends and renders. pytest cannot reach the browser-side code, so
 * this is the only coverage app.js has.
 *
 * It loads the shipped file and drives it — no logic is reimplemented here.
 *
 *   npm install jsdom        # once, anywhere
 *   node tests/web/verify-ui.js [--jsdom-path /path/to/node_modules/jsdom]
 */
const fs = require("fs");
const path = require("path");

function loadJsdom() {
  const arg = process.argv.indexOf("--jsdom-path");
  const candidates = [];
  if (arg > -1) candidates.push(process.argv[arg + 1]);
  candidates.push("jsdom");
  candidates.push(path.join(__dirname, "../../node_modules/jsdom"));
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (_) {
      /* try the next location */
    }
  }
  console.error("jsdom not found. Install it with: npm install jsdom");
  console.error("(or pass --jsdom-path /path/to/node_modules/jsdom)");
  process.exit(77); // 77 = skipped, by convention
}

const { JSDOM } = loadJsdom();

const REPO = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(REPO, "jarvis/web/index.html"), "utf8");
const appJs = fs.readFileSync(path.join(REPO, "jarvis/web/app.js"), "utf8");

const FAKE_KEY = "AQ.Ab8RbrowserDirectTestKey000000000000";

// ------------------------------------------------------------------ harness
let mode = "ok"; // ok | http401 | sseError
const captured = { gemini: null, server: [] };
const results = [];

function check(label, pass, detail = "") {
  results.push({ label, pass: !!pass, detail });
}

/** A Response stand-in whose body streams SSE in awkwardly-sized chunks. */
function sseResponse(events, status = 200) {
  const payload = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  const bytes = new TextEncoder().encode(payload);
  const chunks = [];
  for (let i = 0; i < bytes.length; i += 17) chunks.push(bytes.slice(i, i + 17));
  let index = 0;
  return {
    ok: status >= 200 && status < 300,
    status,
    body: {
      getReader() {
        return {
          async read() {
            if (index >= chunks.length) return { value: undefined, done: true };
            return { value: chunks[index++], done: false };
          },
        };
      },
    },
    async text() {
      return payload;
    },
    async json() {
      return JSON.parse(payload);
    },
  };
}

function jsonResponse(obj, status = 200) {
  const body = JSON.stringify(obj);
  return {
    ok: status >= 200 && status < 300,
    status,
    body: null,
    async text() {
      return body;
    },
    async json() {
      return JSON.parse(body);
    },
  };
}

const tick = (ms = 200) => new Promise((r) => setTimeout(r, ms));

async function boot() {
  const dom = new JSDOM(html, {
    url: "https://jarvis.example/",
    runScripts: "outside-only",
    pretendToBeVisual: true,
  });
  const { window } = dom;
  window.TextDecoder = TextDecoder;
  window.TextEncoder = TextEncoder;

  window.fetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("generativelanguage.googleapis.com")) {
      captured.gemini = { url: u, opts };
      if (mode === "http401") {
        return {
          ok: false,
          status: 401,
          body: null,
          async text() {
            return '{"error":{"code":401,"message":"API key not valid"}}';
          },
        };
      }
      if (mode === "sseError") {
        return sseResponse([
          { error: { code: 429, message: "Quota exceeded for this project" } },
        ]);
      }
      return sseResponse([
        { candidates: [{ content: { parts: [{ text: "Streamed " }] } }] },
        { candidates: [{ content: { parts: [{ text: "from " }] } }] },
        { candidates: [{ content: { parts: [{ text: "Gemini." }] } }] },
      ]);
    }
    captured.server.push({ url: u, opts });
    if (u.endsWith("/api/health")) {
      return jsonResponse({
        status: "ok",
        version: "0.1.0",
        model: "gemini-2.5-flash",
        provider: "google-ai-studio",
        api_key_configured: false,
        api_key_hint: "",
        auth_required: false,
        messages: [],
      });
    }
    if (u.endsWith("/api/key")) {
      return jsonResponse({
        api_key_configured: true,
        api_key_hint: "AQ.Ab8…000",
        model: "gemini-2.5-flash",
        provider: "google-ai-studio",
      });
    }
    if (u.endsWith("/api/model")) {
      return jsonResponse({ model: "gemini-2.5-flash", provider: "google-ai-studio", api_key_configured: false });
    }
    if (u.endsWith("/api/reset")) return jsonResponse({ messages: [] });
    if (u.endsWith("/api/chat")) {
      return sseResponse([
        { type: "start", model: "gemini-2.5-flash", provider: "google-ai-studio" },
        { type: "error", message: "No Google AI Studio API key found." },
      ]);
    }
    return jsonResponse({}, 404);
  };

  window.localStorage.setItem("jarvis.route", "browser");
  window.localStorage.setItem("jarvis.browserApiKey", FAKE_KEY);
  window.eval(appJs);
  await tick(60);
  return window;
}

async function send(window, text) {
  captured.gemini = null;
  const doc = window.document;
  doc.getElementById("input").value = text;
  doc.getElementById("composer").dispatchEvent(
    new window.Event("submit", { bubbles: true, cancelable: true })
  );
  await tick(200);
}

const bubbles = (window, cls) =>
  [...window.document.querySelectorAll(`.msg.${cls}`)].map((m) =>
    m.textContent.replace(/^(Jarvis|You)/, "").trim()
  );

// -------------------------------------------------------------------- tests
async function testHappyPath() {
  const window = await boot();
  await send(window, "hello jarvis");

  const g = captured.gemini;
  check("browser calls the Gemini endpoint directly", !!g, g ? g.url : "no call made");
  if (g) {
    check("uses streamGenerateContent with alt=sse", g.url.includes(":streamGenerateContent?alt=sse"));
    check("model comes from /api/health", g.url.includes("/gemini-2.5-flash:"));
    check(
      "key goes in the x-goog-api-key header, never the URL",
      g.opts.headers["x-goog-api-key"] === FAKE_KEY && !g.url.includes(FAKE_KEY)
    );
    const body = JSON.parse(g.opts.body);
    check("user turn maps to role 'user'", body.contents[0].role === "user");
    check("prompt text is carried through", body.contents[0].parts[0].text === "hello jarvis");
    check("system instruction is included", !!body.systemInstruction?.parts?.[0]?.text);
  }

  check(
    "streamed chunks are reassembled in the assistant bubble",
    bubbles(window, "assistant").at(-1) === "Streamed from Gemini.",
    JSON.stringify(bubbles(window, "assistant").at(-1))
  );
  check("user message is shown", bubbles(window, "user").includes("hello jarvis"));
  check("streaming cursor is cleared when done", !window.document.querySelector(".msg.assistant.cursor"));
  check(
    "status bar reports the direct route",
    window.document.getElementById("status").textContent.includes("browser-direct"),
    window.document.getElementById("status").textContent
  );

  await send(window, "and again");
  const second = captured.gemini ? JSON.parse(captured.gemini.opts.body) : null;
  check(
    "the next turn replays history, with the reply as role 'model'",
    !!second &&
      second.contents.length === 3 &&
      second.contents.map((c) => c.role).join(",") === "user,model,user",
    second ? second.contents.map((c) => c.role).join(",") : "no call"
  );
}

async function testHttpRejection() {
  mode = "http401";
  const window = await boot();
  await send(window, "will this work?");

  const errors = bubbles(window, "error");
  check("an HTTP 401 from Google surfaces as an error bubble", errors.length === 1, JSON.stringify(errors));
  check(
    "the error explains what Google said",
    errors.some((e) => e.includes("401") && e.includes("API key not valid")),
    errors.join(" | ")
  );
  check("no empty assistant bubble is left behind", !bubbles(window, "assistant").includes(""));
  check("send button is re-enabled after the failure", !window.document.getElementById("sendBtn").disabled);
}

async function testSseErrorEvent() {
  mode = "sseError";
  const window = await boot();
  await send(window, "quota check");

  const errors = bubbles(window, "error");
  check("an in-stream error event surfaces as an error bubble", errors.length === 1, JSON.stringify(errors));
  check(
    "the quota message reaches the user",
    errors.some((e) => e.includes("Quota exceeded")),
    errors.join(" | ")
  );
}

async function testServerRouteKeyHandling() {
  mode = "ok";
  const window = await boot();
  const doc = window.document;

  window.localStorage.removeItem("jarvis.browserApiKey");
  doc.getElementById("route").value = "server";
  doc.getElementById("route").dispatchEvent(new window.Event("change"));
  await tick(60);

  doc.getElementById("apiKey").value = FAKE_KEY;
  doc.getElementById("saveKey").dispatchEvent(new window.Event("click", { bubbles: true }));
  await tick(150);

  const keyCall = captured.server.find((c) => String(c.url).endsWith("/api/key"));
  check("server route posts the key to our own API", !!keyCall);
  check(
    "the key is posted in the body, not the URL",
    !!keyCall && JSON.parse(keyCall.opts.body).api_key === FAKE_KEY && !String(keyCall.url).includes(FAKE_KEY)
  );
  check(
    "'remember' off keeps the key out of localStorage",
    !window.localStorage.getItem("jarvis.browserApiKey")
  );
  check(
    "the key goes to sessionStorage instead",
    window.sessionStorage.getItem("jarvis.browserApiKey.session") === FAKE_KEY
  );
  check("the password field is cleared after saving", doc.getElementById("apiKey").value === "");

  // The server reports a missing key over SSE; the UI must show it, not hang.
  await send(window, "hello");
  const errors = bubbles(window, "error");
  check("a server-side SSE error is rendered", errors.some((e) => e.includes("No Google AI Studio API key")), errors.join(" | "));
}

// --------------------------------------------------------------------- main
(async () => {
  for (const test of [testHappyPath, testHttpRejection, testSseErrorEvent, testServerRouteKeyHandling]) {
    await test();
  }
  console.log("CHECK                                                            PASS   DETAIL");
  console.log("-".repeat(88));
  for (const r of results) {
    console.log(`${r.label.padEnd(64)} ${(r.pass ? "yes" : "NO").padEnd(6)} ${r.detail}`);
  }
  const failed = results.filter((r) => !r.pass);
  console.log("-".repeat(88));
  console.log(`${results.length} checks, ${failed.length} failed`);
  process.exit(failed.length ? 1 : 0);
})().catch((err) => {
  console.error("HARNESS ERROR:", err);
  process.exit(2);
});
