import { useEffect, useRef, useState } from "react";

import { api, streamChat } from "../api/client.js";
import ChatMessage from "../components/ChatMessage.jsx";

const EXAMPLES = [
  "Which customers are most likely to churn, and does that correlate with contract type?",
  "What is the churn risk for customer 7590-VHVEG and what drives it?",
  "A senior on a month-to-month fiber contract, tenure 2, paying $95/month — churn risk?",
  "Average churn risk by payment method",
  "Does churn risk correlate with region?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef(null);

  useEffect(() => {
    // braces matter: scrollIntoView returns a Promise in newer Chrome, and an
    // implicit return would hand it to React as an effect-cleanup function
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function ask(question) {
    if (!question.trim() || busy) return;
    setBusy(true);
    setInput("");
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "", events: [], thinking: true },
    ]);

    const update = (patch) =>
      setMessages((m) => {
        const next = [...m];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });

    try {
      for await (const event of streamChat(question)) {
        if (event.type === "result") {
          update({
            content: event.answer,
            verification: event.verification,
            stats: event,
            thinking: false,
          });
        } else if (event.type !== "final" && event.type !== "draft") {
          setMessages((m) => {
            const next = [...m];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, events: [...last.events, event] };
            return next;
          });
        }
      }
    } catch {
      update({
        content:
          "Something went wrong while answering — nothing was fabricated. Please try again.",
        thinking: false,
      });
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    await api.reset().catch(() => {});
    setMessages([]);
  }

  return (
    <div className="page chat-page">
      <header className="page-head">
        <h1>Chat with the analyst</h1>
        <button className="ghost" onClick={reset} disabled={busy}>
          ↺ reset conversation
        </button>
      </header>

      {messages.length === 0 && (
        <div className="examples">
          <p className="examples-title">Try one:</p>
          {EXAMPLES.map((q) => (
            <button key={q} className="chip" onClick={() => ask(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="thread">
        {messages.map((message, i) => (
          <ChatMessage key={i} message={message} />
        ))}
        <div ref={bottom} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the data, a customer's churn risk, or a what-if…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
