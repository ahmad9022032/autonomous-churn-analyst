import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Chat", icon: "💬", end: true },
  { to: "/dataset", label: "Dataset", icon: "🗂️" },
  { to: "/model", label: "Customer Risk", icon: "🎯" },
  { to: "/whatif", label: "What-If Lab", icon: "🧪" },
];

export default function Nav() {
  return (
    <aside className="nav">
      <div className="brand">
        <span className="brand-mark">📉</span>
        <div>
          <div className="brand-name">ChurnSight</div>
          <div className="brand-sub">autonomous churn analyst</div>
        </div>
      </div>
      <nav>
        {LINKS.map(({ to, label, icon, end }) => (
          <NavLink key={to} to={to} end={end} className="nav-link">
            <span className="nav-icon">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>
      <p className="nav-blurb">
        Ask anything about 7,043 telecom customers. The agent plans, computes
        with real tools, self-checks — and every number is verified against an
        actually-computed result.
      </p>
    </aside>
  );
}
