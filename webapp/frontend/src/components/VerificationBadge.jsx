export default function VerificationBadge({ verification, stats }) {
  if (!verification && !stats) return null;
  return (
    <div className="badge-row">
      {verification && (
        <span className={`badge ${verification.ok ? "badge-good" : "badge-bad"}`}>
          {verification.ok ? "✅" : "🚫"} {verification.summary}
        </span>
      )}
      {stats && (
        <span className="badge badge-muted">
          {stats.llm_calls} LLM calls · {stats.tool_steps} tool steps · {stats.elapsed_s}s
        </span>
      )}
    </div>
  );
}
