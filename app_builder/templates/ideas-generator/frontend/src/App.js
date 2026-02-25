import React, { useState, useEffect } from 'react';

const CACHE_KEY = 'ideas_generator_cache';
const MAX_CACHE = 15;

function getCache() {
  try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || []; } catch { return []; }
}
function saveToCache(topic, ideas) {
  const arr = getCache();
  arr.unshift({ topic, ideas, ts: Date.now() });
  localStorage.setItem(CACHE_KEY, JSON.stringify(arr.slice(0, MAX_CACHE)));
}

function getBackendUrl() {
  const env = (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || '').trim();
  if (env) return env;
  if (typeof window !== 'undefined') {
    if (window.__BACKEND_URL__) return window.__BACKEND_URL__;
    return `${window.location.protocol}//${window.location.hostname}:5001`;
  }
  return 'http://localhost:5001';
}

function App() {
  const backendUrl = getBackendUrl();
  const [topic, setTopic] = useState('');
  const [count, setCount] = useState(5);
  const [ideas, setIdeas] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fromCache, setFromCache] = useState(false);
  const [cachedList, setCachedList] = useState([]);

  useEffect(() => { setCachedList(getCache()); }, [ideas]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!topic.trim()) return;
    setLoading(true);
    setError('');
    setIdeas([]);
    setFromCache(false);
    const cacheKey = topic.trim().toLowerCase();
    const cached = getCache().find(c => c.topic?.toLowerCase() === cacheKey);
    if (cached && cached.ideas?.length > 0) {
      setIdeas(cached.ideas);
      setFromCache(true);
      setLoading(false);
      return;
    }
    try {
      const res = await fetch(`${backendUrl}/generate-ideas`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topic.trim(), count }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      setIdeas(data.ideas || []);
      saveToCache(topic.trim(), data.ideas || []);
    } catch (err) {
      setError(
        cachedList.length > 0
          ? 'Backend not running. Cached ideas below – click to view.'
          : 'Backend not running. Add OPENAI_API_KEY to .env and start backend (cd backend && uvicorn main:app --port 5001) for AI ideas.',
      );
      setCachedList(getCache());
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Ideas Generator</h1>
        <p className="app-subtitle">Get creative ideas for any topic using AI</p>
      </header>

      <div className="card">
        <form onSubmit={handleGenerate}>
          <div className="input-group">
            <label className="input-label">Topic or prompt</label>
            <input
              className="input"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. startup ideas, blog post topics, product features..."
              disabled={loading}
            />
          </div>
          <div className="input-group">
            <label className="input-label">Number of ideas</label>
            <select className="input select-input" value={count} onChange={(e) => setCount(parseInt(e.target.value))} disabled={loading}>
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n} ideas</option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading || !topic.trim()}>
            {loading ? 'Generating...' : 'Generate Ideas'}
          </button>
        </form>
      </div>

      {error && (
        <div className="card error-card">
          <p className="error-message">{error}</p>
          {cachedList.length > 0 && (
            <div className="cached-section">
              <p className="cached-label">Recent cached ideas:</p>
              {cachedList.slice(0, 5).map((c, i) => (
                <div key={i} className="cached-item" onClick={() => { setTopic(c.topic); setIdeas(c.ideas || []); setError(''); }}>
                  <span className="cached-topic">{c.topic}</span> – {c.ideas?.length || 0} ideas
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {ideas.length > 0 && (
        <div className="card result-card">
          <h2 className="section-title">Ideas for &quot;{topic}&quot; {fromCache && <span className="badge-cache">(from cache)</span>}</h2>
          <ol className="ideas-list">
            {ideas.map((idea, i) => (
              <li key={i} className="idea-item">{idea}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export default App;
