import { useEffect, useState } from "react";

import { api } from "../api/client.js";
import RiskBar from "../components/RiskBar.jsx";

export default function WhatIfPage() {
  const [schema, setSchema] = useState(null);
  const [customerId, setCustomerId] = useState("7590-VHVEG");
  const [changes, setChanges] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => setError(String(e)));
  }, []);

  function setChange(column, value) {
    setChanges((c) => {
      const next = { ...c };
      if (value === "") delete next[column];
      else next[column] = value;
      return next;
    });
  }

  async function run(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.whatIf(customerId.trim(), changes));
    } catch (err) {
      setError(String(err.message ?? err));
    } finally {
      setBusy(false);
    }
  }

  const data = result?.data;
  const delta = data ? data.projected_risk - data.baseline_risk : 0;

  return (
    <div className="page">
      <header className="page-head"><h1>What-If Lab</h1></header>
      <p className="muted">
        Project how an existing customer's churn risk would change under different
        feature values — both scores come from the same served model, and structural
        consistency (phone/internet dependencies) is enforced automatically.
      </p>

      <form className="whatif" onSubmit={run}>
        <label>
          Customer ID
          <input value={customerId} onChange={(e) => setCustomerId(e.target.value)} />
        </label>

        {schema && (
          <div className="whatif-grid">
            {Object.entries(schema.categorical).map(([column, values]) => (
              <label key={column}>
                {column}
                <select value={changes[column] ?? ""} onChange={(e) => setChange(column, e.target.value)}>
                  <option value="">— keep current —</option>
                  {values.map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </label>
            ))}
            {schema.numeric.map((column) => (
              <label key={column}>
                {column}
                <input
                  type="number"
                  step="any"
                  placeholder="keep current"
                  value={changes[column] ?? ""}
                  onChange={(e) => setChange(column, e.target.value === "" ? "" : Number(e.target.value))}
                />
              </label>
            ))}
          </div>
        )}

        <button type="submit" disabled={busy || Object.keys(changes).length === 0}>
          {busy ? "…" : `Project risk (${Object.keys(changes).length} change${Object.keys(changes).length === 1 ? "" : "s"})`}
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {data && (
        <div className="card">
          <RiskBar score={data.baseline_risk} label="Baseline (today)" />
          <RiskBar score={data.projected_risk} label="Projected (with your changes)" />
          <p className={`delta ${delta < 0 ? "down" : "up"}`}>
            {delta < 0 ? "▼" : "▲"} risk {delta < 0 ? "drops" : "rises"} by{" "}
            <b>{Math.abs(delta).toFixed(4)}</b> ({Math.abs(delta * 100).toFixed(1)} percentage points)
          </p>
          {data.consistency_fixes.length > 0 && (
            <p className="muted">auto-consistency: {data.consistency_fixes.join("; ")}</p>
          )}
        </div>
      )}
    </div>
  );
}
