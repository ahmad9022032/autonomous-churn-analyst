// Live trace of the agent working: plan, tool calls, self-checks, verdict.
// Rendered inside an assistant message while (and after) it thinks.

const ICONS = { ok: "✅", empty: "⚠️", error: "❌" };

function line(event, i) {
  switch (event.type) {
    case "plan":
      return (
        <div className="trace-line trace-plan" key={i}>
          <b>PLAN</b> {event.plan}
        </div>
      );
    case "tool_call":
      return (
        <div className="trace-line" key={i}>
          🔧 <code>{event.tool}({JSON.stringify(event.args).slice(0, 90)})</code>
        </div>
      );
    case "tool_result":
      return (
        <div className="trace-line" key={i}>
          {ICONS[event.status] ?? "⚠️"} {event.status}: <code>{event.summary?.slice(0, 100)}</code>
        </div>
      );
    case "self_check_retry":
      return (
        <div className="trace-line trace-warn" key={i}>
          ⚠️ self-check: {event.note}
        </div>
      );
    case "revision":
      return (
        <div className="trace-line trace-warn" key={i}>
          🚫 verification rejected draft (unverified: {event.unmatched?.join(", ")}) — revising
        </div>
      );
    case "verify_verdict":
      return (
        <div className={`trace-line ${event.ok ? "trace-good" : "trace-bad"}`} key={i}>
          🔍 {event.summary}
        </div>
      );
    default:
      return null;
  }
}

export default function EventTrace({ events, open }) {
  const lines = events.map(line).filter(Boolean);
  if (lines.length === 0) return null;
  return (
    <details className="trace" open={open}>
      <summary>how I worked — {events.filter((e) => e.type === "tool_call").length} tool call(s)</summary>
      {lines}
    </details>
  );
}
