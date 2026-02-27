import React, { useState, useEffect } from 'react';

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'zh-cn', name: 'Chinese (Simplified)' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
];

const CACHE_KEY = 'translator_cache';
const MAX_CACHE = 20;

function getCache() {
  try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || []; } catch { return []; }
}
function saveToCache(entry) {
  const arr = getCache();
  arr.unshift({ ...entry, ts: Date.now() });
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
  const [text, setText] = useState('');
  const [sourceLang, setSourceLang] = useState('auto');
  const [targetLang, setTargetLang] = useState('es');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cacheHint, setCacheHint] = useState(false);
  const [cachedList, setCachedList] = useState([]);

  useEffect(() => { setCachedList(getCache()); }, [result]);

  const handleTranslate = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    setCacheHint(false);
    const cacheKey = `${text.trim()}|${sourceLang}|${targetLang}`;
    const cached = getCache().find(c => `${c.original_text}|${c.source_lang}|${c.target_lang}` === cacheKey);
    if (cached) {
      setResult({ original_text: cached.original_text, translated_text: cached.translated_text, source_lang: cached.source_lang, target_lang: cached.target_lang });
      setCacheHint(true);
      setLoading(false);
      return;
    }
    try {
      const res = await fetch(`${backendUrl}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), source_lang: sourceLang === 'auto' ? 'auto' : sourceLang, target_lang: targetLang }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      setResult(data);
      saveToCache(data);
    } catch (err) {
      const cache = getCache();
      setCachedList(cache);
      setError(cache.length > 0
        ? 'Backend not running. Cached translations below – click to view.'
        : 'Backend not running. Start backend (cd backend && uvicorn main:app --port 5001) for live translation.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Language Translator</h1>
        <p className="app-subtitle">Translate text between 70+ languages</p>
      </header>

      <div className="card translate-card">
        <form onSubmit={handleTranslate}>
          <div className="input-group">
            <label className="input-label">Text to translate</label>
            <textarea
              className="input textarea-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text..."
              rows={4}
              disabled={loading}
            />
          </div>
          <div className="form-grid">
            <div className="input-group">
              <label className="input-label">From language</label>
              <select
                className="input select-input"
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value)}
                disabled={loading}
              >
                <option value="auto">Auto-detect</option>
                {LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>{l.name}</option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label className="input-label">To language</label>
              <select
                className="input select-input"
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                disabled={loading}
              >
                {LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>{l.name}</option>
                ))}
              </select>
            </div>
          </div>
          <button className="btn btn-primary btn-translate" type="submit" disabled={loading || !text.trim()}>
            {loading ? 'Translating...' : 'Translate'}
          </button>
        </form>
      </div>

      {error && (
        <div className="card error-card">
          <p className="error-message">{error}</p>
          {cachedList.length > 0 && (
            <div className="cached-section">
              <p className="cached-label">Recent cached translations:</p>
              {cachedList.slice(0, 5).map((c, i) => (
                <div key={i} className="cached-item" onClick={() => setResult({ original_text: c.original_text, translated_text: c.translated_text, source_lang: c.source_lang, target_lang: c.target_lang })}>
                  <span className="cached-orig">{c.original_text?.slice(0, 40)}{(c.original_text?.length || 0) > 40 ? '...' : ''}</span> → <span className="cached-trans">{c.translated_text?.slice(0, 40)}{(c.translated_text?.length || 0) > 40 ? '...' : ''}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="card result-card">
          <h2 className="section-title">Translation {cacheHint && <span className="badge-cache">(from cache)</span>}</h2>
          <div className="result-content">
            <div className="result-block">
              <span className="result-label">Original</span>
              <p className="result-text">{result.original_text}</p>
            </div>
            <div className="result-block">
              <span className="result-label">Translated</span>
              <p className="result-text translated">{result.translated_text}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
