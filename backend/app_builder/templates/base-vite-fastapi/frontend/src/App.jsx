import { useEffect, useState } from 'react';

/**
 * Generic shell — replaced entirely by codegen from PRD/UI-UX/App Spec.
 * Do not add domain features here.
 */
export default function App() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setReady(true);
  }, []);

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <h1 className="app-title">Generated App</h1>
          <p className="app-subtitle">
            {ready ? 'Waiting for application UI from code generation…' : 'Loading…'}
          </p>
        </header>
        <main className="card shell-notice">
          <p className="muted">
            This is the generic Vite + React shell. Run code generation to build your app UI.
          </p>
        </main>
      </div>
    </div>
  );
}
