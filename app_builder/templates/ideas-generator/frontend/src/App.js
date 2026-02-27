import React, { useState, useEffect } from 'react';

// Tailor can change GEN_TYPE: "ideas" | "linkedin_post" | "travel" | "translate" | "general"
const GEN_TYPE = 'ideas';
const CACHE_KEY = 'llm_generator_cache';
const MAX_CACHE = 15;

function getCache() {
  try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || []; } catch { return []; }
}
function saveToCache(topic, content, genType) {
  const arr = getCache();
  arr.unshift({ topic, content, genType, ts: Date.now() });
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

const LABELS = {
  ideas: { title: 'Ideas Generator', subtitle: 'Get creative ideas for any topic using AI', placeholder: 'e.g. startup ideas, blog post topics...', countLabel: 'Number of ideas', generateBtn: 'Generate Ideas', resultTitle: 'Ideas for' },
  linkedin_post: { title: 'LinkedIn Post Generator', subtitle: 'Generate professional LinkedIn posts using AI', placeholder: 'e.g. Tips for remote work, My journey...', countLabel: null, generateBtn: 'Generate Post', resultTitle: 'LinkedIn Post for' },
  travel: { title: 'Travel Ideas', subtitle: 'Get travel suggestions and tips using AI', placeholder: 'e.g. Tokyo 3 days, budget Europe...', countLabel: 'Number of suggestions', generateBtn: 'Generate', resultTitle: 'Travel ideas for' },
  translate: { title: 'Translator', subtitle: 'Translate text using AI', placeholder: 'Enter text to translate...', countLabel: null, generateBtn: 'Translate', resultTitle: 'Translation' },
  general: { title: 'AI Content Generator', subtitle: 'Generate content for any topic', placeholder: 'Enter your topic or prompt...', countLabel: 'Number of items', generateBtn: 'Generate', resultTitle: 'Results for' },
};

function App() {
  const backendUrl = getBackendUrl();
  const genType = GEN_TYPE;
  const labels = LABELS[genType] || LABELS.ideas;
  const showCount = genType !== 'linkedin_post' && genType !== 'translate';

  const [topic, setTopic] = useState('');
  const [count, setCount] = useState(5);
  const [content, setContent] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fromCache, setFromCache] = useState(false);
  const [cachedList, setCachedList] = useState([]);

  useEffect(() => { setCachedList(getCache()); }, [content]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!topic.trim()) return;
    setLoading(true);
    setError('');
    setContent([]);
    setFromCache(false);
    const cacheKey = topic.trim().toLowerCase();
    const cached = getCache().find(c => c.topic?.toLowerCase() === cacheKey && (c.genType || 'ideas') === genType);
    if (cached && cached.content?.length > 0) {
      setContent(cached.content);
      setFromCache(true);
      setLoading(false);
      return;
    }
    try {
      const res = await fetch(`${backendUrl}/generate-ideas`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topic.trim(), count: showCount ? count : 1, gen_type: genType }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      const items = data.content || [];
      setContent(items);
      saveToCache(topic.trim(), items, genType);
    } catch (err) {
      setError(
        cachedList.length > 0
          ? 'Backend not running. Cached results below – click to view.'
          : 'Backend not running. Add OPENAI_API_KEY to .env and start backend (cd backend && uvicorn main:app --port 5001) for AI generation.',
      );
      setCachedList(getCache());
    } finally {
      setLoading(false);
    }
  };

  const isSingleBlock = genType === 'linkedin_post' || (genType === 'translate' && content.length === 1);

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">{labels.title}</h1>
        <p className="app-subtitle">{labels.subtitle}</p>
      </header>

      <div className="card">
        <form onSubmit={handleGenerate}>
          <div className="input-group">
            <label className="input-label">{genType === 'translate' ? 'Text to translate' : 'Topic or prompt'}</label>
            <input
              className="input"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder={labels.placeholder}
              disabled={loading}
            />
          </div>
          {showCount && (
            <div className="input-group">
              <label className="input-label">{labels.countLabel}</label>
              <select className="input select-input" value={count} onChange={(e) => setCount(parseInt(e.target.value))} disabled={loading}>
                {[3, 5, 7, 10].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
          )}
          <button className="btn btn-primary" type="submit" disabled={loading || !topic.trim()}>
            {loading ? 'Generating...' : labels.generateBtn}
          </button>
        </form>
      </div>

      {error && (
        <div className="card error-card">
          <p className="error-message">{error}</p>
          {cachedList.length > 0 && (
            <div className="cached-section">
              <p className="cached-label">Recent cached:</p>
              {cachedList.slice(0, 5).map((c, i) => (
                <div key={i} className="cached-item" onClick={() => { setTopic(c.topic); setContent(c.content || []); setError(''); }}>
                  <span className="cached-topic">{c.topic}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {content.length > 0 && (
        <div className="card result-card">
          <h2 className="section-title">{labels.resultTitle} &quot;{topic}&quot; {fromCache && <span className="badge-cache">(from cache)</span>}</h2>
          {isSingleBlock ? (
            <div className="post-content">{content[0]}</div>
          ) : (
            <ol className="ideas-list">
              {content.map((item, i) => (
                <li key={i} className="idea-item">{item}</li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
