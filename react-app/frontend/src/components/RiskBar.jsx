// Risk score as a horizontal meter with the percentile in context.
export default function RiskBar({ score, percentile, label }) {
  const pct = Math.round(score * 100);
  const tone = score >= 0.5 ? "risk-high" : score >= 0.25 ? "risk-mid" : "risk-low";
  return (
    <div className="riskbar">
      {label && <div className="riskbar-label">{label}</div>}
      <div className="riskbar-track">
        <div className={`riskbar-fill ${tone}`} style={{ width: `${Math.max(pct, 2)}%` }} />
      </div>
      <div className="riskbar-meta">
        <b>{score.toFixed(4)}</b> churn probability
        {percentile != null && <> · riskier than {Math.round(percentile * 100)}% of customers</>}
      </div>
    </div>
  );
}
