import React, { useState, useEffect } from 'react';

const STORAGE_KEY = 'todos_app';

function TodoManager({ backendUrl }) {
  const [items, setItems] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; } catch { return []; }
  });
  const [formData, setFormData] = useState({ title: '', description: '', priority: 'medium' });
  const [filter, setFilter] = useState('all');

  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(items)); }, [items]);

  const fetchData = async () => {
    try {
      const r = await fetch(`${backendUrl}/todos`);
      if (r.ok) { const d = await r.json(); setItems(Array.isArray(d) ? d : []); }
    } catch (e) {}
  };
  useEffect(() => { fetchData(); }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const newItem = { ...formData, completed: false, id: Date.now() };
    setItems([...items, newItem]);
    try {
      await fetch(`${backendUrl}/todos`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formData) });
      fetchData();
    } catch (e) {}
    setFormData({ title: '', description: '', priority: 'medium' });
  };

  const toggleComplete = async (item) => {
    const updated = { ...item, completed: !item.completed };
    setItems(items.map(i => i.id === item.id ? updated : i));
    try {
      await fetch(`${backendUrl}/todos/${item.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(updated) });
    } catch (e) {}
  };

  const handleDelete = async (id) => {
    setItems(items.filter(i => i.id !== id));
    try {
      await fetch(`${backendUrl}/todos/${id}`, { method: 'DELETE' });
    } catch (e) {}
  };

  const filtered = filter === 'active' ? items.filter(i => !i.completed) : filter === 'completed' ? items.filter(i => i.completed) : items;

  return (
    <div className="section">
      <h2 className="section-title">My Tasks</h2>
      <div className="card todo-card">
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="input-group">
              <label className="input-label">Title</label>
              <input className="input" value={formData.title} onChange={e => setFormData({ ...formData, title: e.target.value })} placeholder="What needs to be done?" required />
            </div>
            <div className="input-group">
              <label className="input-label">Priority</label>
              <select className="input" value={formData.priority} onChange={e => setFormData({ ...formData, priority: e.target.value })}>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
          <div className="input-group">
            <label className="input-label">Description (optional)</label>
            <input className="input" value={formData.description} onChange={e => setFormData({ ...formData, description: e.target.value })} placeholder="Add details..." />
          </div>
          <button className="btn btn-primary" type="submit">Add Task</button>
        </form>
      </div>
      <div className="filter-tabs">
        {['all', 'active', 'completed'].map(f => (
          <button key={f} className={`filter-tab ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>
      <div className="list">
        {filtered.map(item => (
          <div key={item.id} className={`list-item todo-item ${item.completed ? 'todo-item-completed' : ''}`}>
            <div className="todo-item-left">
              <input type="checkbox" className="todo-checkbox" checked={!!item.completed} onChange={() => toggleComplete(item)} />
              <div>
                <div className={`todo-text ${item.completed ? 'completed' : ''}`}>{item.title}</div>
                {item.description && <div className="todo-desc">{item.description}</div>}
                <span className={`todo-badge todo-badge-${(item.priority || 'medium').toLowerCase()}`}>{item.priority || 'medium'}</span>
              </div>
            </div>
            <button className="btn btn-danger" onClick={() => handleDelete(item.id)}>Delete</button>
          </div>
        ))}
      </div>
    </div>
  );
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
  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Todo App</h1>
        <p className="app-subtitle">Organize your tasks with priority and filters</p>
      </header>
      <TodoManager backendUrl={backendUrl} />
    </div>
  );
}

export default App;
