/* Jarvis web client: streaming chat over SSE, via this server or direct to Google. */
(() => {
  "use strict";

  const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models";
  const LS_KEY = "jarvis.browserApiKey";
  const SS_KEY = "jarvis.browserApiKey.session";
  const SS_TOKEN = "jarvis.authToken";
  const LS_ROUTE = "jarvis.route";
  const LS_MODEL = "jarvis.model";

  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const statusEl = document.getElementById("status");
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsPanel = document.getElementById("settingsPanel");
  const routeSel = document.getElementById("route");
  const keyInput = document.getElementById("apiKey");
  const keyHint = document.getElementById("keyHint");
  const saveKeyBtn = document.getElementById("saveKey");
  const modelInput = document.getElementById("model");
  const tokenInput = document.getElementById("authToken");
  const rememberKey = document.getElementById("rememberKey");
  const resetBtn = document.getElementById("resetBtn");

  let serverState = { model: "gemini-2.5-flash", provider: "google-ai-studio", api_key_configured: false };
  let transcript = []; // [{role, text}] used by the browser-direct route
  let busy = false;

  // --------------------------------------------------------------------- secrets
  const readKey = () => (localStorage.getItem(LS_KEY) || sessionStorage.getItem(SS_KEY) || "").trim();

  function storeKey(key) {
    if (rememberKey.checked) {
      localStorage.setItem(LS_KEY, key);
      sessionStorage.removeItem(SS_KEY);
    } else {
      sessionStorage.setItem(SS_KEY, key);
      localStorage.removeItem(LS_KEY); // never leave a copy behind
    }
  }

  const readToken = () => sessionStorage.getItem(SS_TOKEN) || "";

  /** Headers for calls to our own server, including the optional auth token. */
  function serverHeaders() {
    const headers = { "Content-Type": "application/json" };
    const token = readToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    return headers;
  }

  // --------------------------------------------------------------------- helpers
  const setStatus = (text, cls = "") => {
    statusEl.textContent = text;
    statusEl.className = `status ${cls}`;
  };

  function addMessage(role, text = "") {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = role === "user" ? "You" : role === "assistant" ? "Jarvis" : "";
    el.appendChild(who);
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text;
    el.appendChild(body);
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return body;
  }

  /** Iterate `data: {...}` payloads from a streaming fetch response. */
  async function* sseEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop();
      for (const block of blocks) {
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          try {
            yield JSON.parse(payload);
          } catch (_) {
            /* ignore keep-alive noise */
          }
        }
      }
    }
  }

  // ---------------------------------------------------------------- server route
  async function streamViaServer(message, sink) {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: serverHeaders(),
      body: JSON.stringify({ message }),
    });
    if (response.status === 401) {
      throw new Error("This server requires a token. Open Settings and set the server token.");
    }
    if (!response.ok || !response.body) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Server error ${response.status} ${detail}`.trim());
    }
    for await (const evt of sseEvents(response)) {
      if (evt.type === "token") sink(evt.text || "");
      else if (evt.type === "error") throw new Error(evt.message || "Unknown server error");
    }
  }

  // --------------------------------------------------------------- direct route
  async function streamViaBrowser(message, sink) {
    const key = readKey();
    if (!key) throw new Error("No API key saved in this browser. Open Settings and paste one.");
    const model = serverState.model || "gemini-2.5-flash";
    const contents = transcript.concat([{ role: "user", parts: [{ text: message }] }]).map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: m.parts || [{ text: m.text }],
    }));
    const response = await fetch(`${GEMINI_ENDPOINT}/${model}:streamGenerateContent?alt=sse`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({
        contents,
        systemInstruction: { parts: [{ text: "You are Jarvis, a sharp and concise assistant." }] },
        generationConfig: { temperature: 0.7 },
      }),
    });
    if (!response.ok || !response.body) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Google API error ${response.status}: ${detail.slice(0, 300)}`);
    }
    for await (const evt of sseEvents(response)) {
      if (evt.error) throw new Error(evt.error.message || JSON.stringify(evt.error));
      const parts = evt?.candidates?.[0]?.content?.parts || [];
      for (const part of parts) if (part.text) sink(part.text);
    }
  }

  // ---------------------------------------------------------------------- submit
  async function send() {
    const message = input.value.trim();
    if (!message || busy) return;
    busy = true;
    sendBtn.disabled = true;
    input.value = "";
    autosize();

    addMessage("user", message);
    const sink = addMessage("assistant");
    sink.parentElement.classList.add("cursor");
    const chunks = [];

    try {
      const stream = routeSel.value === "browser" ? streamViaBrowser : streamViaServer;
      await stream(message, (text) => {
        chunks.push(text);
        sink.textContent = chunks.join("");
        chat.scrollTop = chat.scrollHeight;
      });
      if (!chunks.length) sink.textContent = "(empty response)";
      transcript.push({ role: "user", text: message }, { role: "assistant", text: chunks.join("") });
      setStatus(`ready · ${serverState.model} · ${routeSel.value === "browser" ? "browser-direct" : "server"}`, "ok");
    } catch (err) {
      sink.parentElement.remove();
      addMessage("error", String(err.message || err));
      setStatus("error on last request", "err");
    } finally {
      sink.parentElement.classList.remove("cursor");
      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  // --------------------------------------------------------------------- config
  async function refreshHealth() {
    try {
      const res = await fetch("/api/health");
      serverState = await res.json();
      if (!modelInput.value) modelInput.value = localStorage.getItem(LS_MODEL) || serverState.model;
      const route = routeSel.value === "browser" ? "browser-direct" : "server";
      const keyed = routeSel.value === "browser" ? !!readKey() : serverState.api_key_configured;
      const auth = serverState.auth_required ? " · token required" : "";
      setStatus(
        `${serverState.model} · ${serverState.provider} · ${route} · ${keyed ? "key set" : "NO KEY"}${auth}`,
        keyed ? "ok" : "warn"
      );
      if (!tokenInput.value) tokenInput.value = readToken();
      rememberKey.checked = !!localStorage.getItem(LS_KEY);
      keyHint.textContent = keyed
        ? `A key is configured for this route${
            serverState.api_key_hint ? ` (${serverState.api_key_hint})` : ""
          }.`
        : "No key yet for this route — paste one below.";
      if (serverState.messages && serverState.messages.length && !chat.childElementCount) {
        for (const m of serverState.messages) addMessage(m.role, m.text);
        transcript = serverState.messages.map((m) => ({ role: m.role, text: m.text }));
      }
    } catch (err) {
      setStatus(`cannot reach server: ${err.message}`, "err");
    }
  }

  async function saveKey() {
    const key = keyInput.value.trim();
    const model = modelInput.value.trim();
    const token = tokenInput.value.trim();
    if (token) sessionStorage.setItem(SS_TOKEN, token);
    storeKey(key);
    localStorage.setItem(LS_MODEL, model || serverState.model);

    if (routeSel.value === "browser") {
      keyInput.value = "";
      tokenInput.value = "";
      await refreshHealth();
      return;
    }
    try {
      const post = (url, body) =>
        fetch(url, { method: "POST", headers: serverHeaders(), body: JSON.stringify(body) }).then(
          async (r) => {
            if (r.status === 401) throw new Error("wrong server token (401)");
            return r.json();
          }
        );
      const results = [await post("/api/key", { api_key: key })];
      if (model) results.push(await post("/api/model", { model }));
      for (const data of results) serverState = { ...serverState, ...data };
      keyInput.value = "";
      tokenInput.value = "";
      await refreshHealth();
    } catch (err) {
      keyHint.textContent = `Could not save: ${err.message}`;
    }
  }

  // ------------------------------------------------------------------- wiring
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  settingsBtn.addEventListener("click", () => settingsPanel.classList.toggle("hidden"));
  saveKeyBtn.addEventListener("click", saveKey);
  routeSel.addEventListener("change", () => {
    localStorage.setItem(LS_ROUTE, routeSel.value);
    refreshHealth();
  });
  resetBtn.addEventListener("click", async () => {
    transcript = [];
    chat.innerHTML = "";
    try {
      await fetch("/api/reset", { method: "POST", headers: serverHeaders() });
    } catch (_) {
      /* direct route still works without the server */
    }
    refreshHealth();
  });

  routeSel.value = localStorage.getItem(LS_ROUTE) || "server";
  refreshHealth();
  input.focus();
})();
