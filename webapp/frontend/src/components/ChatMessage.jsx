import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import EventTrace from "./EventTrace.jsx";
import VerificationBadge from "./VerificationBadge.jsx";

export default function ChatMessage({ message }) {
  const { role, content, events, verification, stats, thinking } = message;
  return (
    <div className={`msg msg-${role}`}>
      <div className="msg-avatar">{role === "user" ? "🧑" : "📉"}</div>
      <div className="msg-body">
        {events?.length > 0 && <EventTrace events={events} open={thinking} />}
        {thinking && !content && <div className="thinking">thinking…</div>}
        {content && (
          <div className="msg-content">
            <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
          </div>
        )}
        <VerificationBadge verification={verification} stats={stats} />
      </div>
    </div>
  );
}
