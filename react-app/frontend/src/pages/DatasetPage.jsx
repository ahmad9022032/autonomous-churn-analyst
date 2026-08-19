import { useEffect, useState } from "react";

import { api } from "../api/client.js";
import StatTile from "../components/StatTile.jsx";

function CategoryCard({ name, values }) {
  const total = Object.values(values).reduce((a, b) => a + b, 0);
  const entries = Object.entries(values).sort((a, b) => b[1] - a[1]);
  return (
    <div className="card">
      <h3>{name}</h3>
      {entries.map(([value, count]) => (
        <div className="hbar" key={value}>
          <span className="hbar-label" title={value}>{value}</span>
          <span className="hbar-track">
            <span className="hbar-fill" style={{ width: `${(count / total) * 100}%` }} />
          </span>
          <span className="hbar-count">{count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

export default function DatasetPage() {
  const [overview, setOverview] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.overview().then(setOverview).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="page"><p className="error">{error}</p></div>;
  if (!overview) return <div className="page"><p className="muted">loading dataset…</p></div>;

  const columns = Object.entries(overview.columns);
  const categorical = columns.filter(([, c]) => c.type === "categorical");
  const numeric = columns.filter(([, c]) => c.type === "numeric");

  return (
    <div className="page">
      <header className="page-head"><h1>The dataset</h1></header>

      <div className="tiles">
        <StatTile label="customers" value={overview.rows.toLocaleString()} />
        <StatTile label="churn rate" value={`${(overview.churn_rate * 100).toFixed(1)}%`} />
        <StatTile label="columns" value={columns.length} />
      </div>

      <div className="card">
        <h3>Data notes — what was found and fixed</h3>
        <ul className="notes">
          {overview.notes.issues_found_and_fixed.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
          {overview.notes.checked_clean.map((note, i) => (
            <li key={`c${i}`} className="muted">✓ {note}</li>
          ))}
        </ul>
      </div>

      <h2>Numeric columns</h2>
      <div className="tiles">
        {numeric.map(([name, c]) => (
          <StatTile
            key={name}
            label={name}
            value={c.mean.toFixed(1)}
            hint={`mean · range ${c.min}–${c.max}`}
          />
        ))}
      </div>

      <h2>Categorical columns</h2>
      <div className="cards">
        {categorical.map(([name, c]) => (
          <CategoryCard key={name} name={name} values={c.values} />
        ))}
      </div>
    </div>
  );
}
