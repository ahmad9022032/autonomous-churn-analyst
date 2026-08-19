import { useEffect, useState } from "react";

import { api } from "../api/client.js";
import RiskBar from "../components/RiskBar.jsx";
import StatTile from "../components/StatTile.jsx";

export default function ModelPage() {
  const [metrics, setMetrics] = useState(null);
  const [customerId, setCustomerId] = useState("7590-VHVEG");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => {});
  }, []);

  async function lookup(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.customer(customerId.trim()));
    } catch (err) {
      setError(String(err.message ?? err));
    } finally {
      setBusy(false);
    }
  }

  const data = result?.data;
  return (
    <div className="page">
      <header className="page-head"><h1>Customer risk lookup</h1></header>

      {metrics && (
        <div className="tiles">
          <StatTile label="PR-AUC (holdout)" value={metrics["PR-AUC"]} />
          <StatTile label="ROC-AUC" value={metrics["ROC-AUC"]} />
          <StatTile label="precision @ top-10%" value={metrics["precision@top-10%"]} />
          <StatTile label="lift @ top-10%" value={`${metrics["lift@top-10%"]}×`} />
        </div>
      )}

      <form className="lookup" onSubmit={lookup}>
        <input
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          placeholder="customer ID, e.g. 7590-VHVEG"
        />
        <button type="submit" disabled={busy || !customerId.trim()}>
          {busy ? "…" : "Score customer"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {data && (
        <div className="card">
          <h3>{customerId.trim()}</h3>
          <RiskBar score={data.risk_score} percentile={data.risk_percentile} />
          <h4>Top factors</h4>
          <ul className="factors">
            {data.top_factors.map((f) => (
              <li key={f.feature} className={f.direction.startsWith("increases") ? "up" : "down"}>
                <span className="factor-arrow">{f.direction.startsWith("increases") ? "▲" : "▼"}</span>
                <b>{f.feature}</b> = {String(f.customer_value)}
                <span className="muted"> · {f.direction} (log-odds {f.log_odds > 0 ? "+" : ""}{f.log_odds})</span>
              </li>
            ))}
          </ul>
          <h4>Snapshot</h4>
          <div className="kv">
            {Object.entries(data.snapshot).map(([key, value]) => (
              <div key={key}><span className="muted">{key}</span> {String(value)}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
