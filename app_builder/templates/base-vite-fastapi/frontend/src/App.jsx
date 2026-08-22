import { useEffect, useState } from 'react';
import { apiFetch } from './api/client.js';

const STORAGE_KEY = 'app_items';

export default function App() {
  const [items, setItems] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
      return [];
    }
  });
  const [title, setTitle] = useState('');
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  }, [items]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await apiFetch('/items');
        if (!cancelled && Array.isArray(data)) setItems(data);
      } catch {
        /* localStorage fallback */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const addItem = async (e) => {
    e.preventDefault();
    const text = title.trim();
    if (!text) return;
    const optimistic = { id: Date.now(), title: text, completed: false };
    setItems((prev) => [optimistic, ...prev]);
    setTitle('');
    try {
      const created = await apiFetch('/items', {
        method: 'POST',
        body: JSON.stringify({ title: text, completed: false }),
      });
      if (created?.id) {
        setItems((prev) => prev.map((i) => (i.id === optimistic.id ? created : i)));
      }
    } catch {
      /* keep optimistic */
    }
  };

  const toggleItem = async (id) => {
    setItems((prev) => prev.map((i) => (i.id === id ? { ...i, completed: !i.completed } : i)));
    const item = items.find((i) => i.id === id);
    if (!item) return;
    try {
      await apiFetch(`/items/${id}`, {
        method: 'PATCH',
        body: JSON.stringify({ completed: !item.completed }),
      });
    } catch {
      /* local state already updated */
    }
  };

  const removeItem = async (id) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
    try {
      await apiFetch(`/items/${id}`, { method: 'DELETE' });
    } catch {
      /* already removed locally */
    }
  };

  const visible = (items || []).filter((i) => {
    if (filter === 'active') return !i.completed;
    if (filter === 'completed') return i.completed;
    return true;
  });

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <h1 className="app-title">My App</h1>
          <p className="app-subtitle">Vite + React + FastAPI scaffold</p>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <div className="card">
          <form className="form-row" onSubmit={addItem}>
            <input
              className="input"
              placeholder="Add an item..."
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <button type="submit" className="btn btn-primary" disabled={!title.trim()}>
              Add
            </button>
          </form>

          <div className="filters">
            {['all', 'active', 'completed'].map((f) => (
              <button
                key={f}
                type="button"
                className={`btn btn-ghost${filter === f ? ' active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>

          {loading ? (
            <p className="empty">Loading...</p>
          ) : visible.length === 0 ? (
            <p className="empty">No items yet. Add one above.</p>
          ) : (
            <ul className="list">
              {visible.map((item) => (
                <li key={item.id} className={`list-item${item.completed ? ' done' : ''}`}>
                  <input
                    type="checkbox"
                    checked={!!item.completed}
                    onChange={() => toggleItem(item.id)}
                  />
                  <span className="item-title" style={{ flex: 1 }}>{item.title}</span>
                  <button type="button" className="btn btn-ghost" onClick={() => removeItem(item.id)}>
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
