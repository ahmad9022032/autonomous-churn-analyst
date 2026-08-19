// Single API surface for the app. The chat endpoint streams NDJSON lines —
// each an agent event (plan / tool_call / tool_result / self_check_retry /
// revision / verify_verdict) — ending with a {type: "result"} line.

const SESSION_KEY = "churnsight-session";

export function sessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`);
  return res.json();
}

export const api = {
  health: () => getJSON("/api/health"),
  overview: () => getJSON("/api/overview"),
  metrics: () => getJSON("/api/metrics"),
  schema: () => getJSON("/api/schema"),
  customer: (id) => getJSON(`/api/customers/${encodeURIComponent(id)}`),
  whatIf: (customer_id, changes) => postJSON("/api/whatif", { customer_id, changes }),
  reset: () => postJSON("/api/reset", { session_id: sessionId() }),
};

export async function* streamChat(question) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId(), question }),
  });
  if (!res.ok || !res.body) throw new Error(`chat request failed (HTTP ${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}
