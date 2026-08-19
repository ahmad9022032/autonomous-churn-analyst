import { Component } from "react";

// A render error must never leave the user staring at a blank page.
export default class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{ padding: 40, fontFamily: "system-ui" }}>
        <h1 style={{ color: "#0b0b0b" }}>Something broke in the interface</h1>
        <p style={{ color: "#52514e" }}>
          The analysis engine is unaffected — this is a display error. Try
          reloading the page.
        </p>
        <pre style={{ background: "#f0efec", padding: 12, borderRadius: 8, overflow: "auto" }}>
          {String(this.state.error)}
        </pre>
      </div>
    );
  }
}
