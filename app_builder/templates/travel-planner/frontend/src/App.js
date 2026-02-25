import React, { useState, useEffect } from 'react';

const STORAGE_TRIPS = 'travel_trips';
const STORAGE_ITINERARY = 'travel_itinerary';

function formatDate(d) {
  if (!d) return '';
  const x = new Date(d);
  return isNaN(x) ? d : x.toLocaleDateString();
}

function TripManager({ backendUrl }) {
  const [trips, setTrips] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_TRIPS)) || []; } catch { return []; }
  });
  const [itinerary, setItinerary] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_ITINERARY)) || []; } catch { return []; }
  });
  const [tripForm, setTripForm] = useState({ title: '', destination: '', start_date: '', end_date: '', budget: '', notes: '' });
  const [itemForm, setItemForm] = useState({ trip_id: '', day: 1, title: '', description: '', time_slot: '' });
  const [selectedTrip, setSelectedTrip] = useState(null);

  useEffect(() => { localStorage.setItem(STORAGE_TRIPS, JSON.stringify(trips)); }, [trips]);
  useEffect(() => { localStorage.setItem(STORAGE_ITINERARY, JSON.stringify(itinerary)); }, [itinerary]);

  const fetchTrips = async () => {
    try {
      const r = await fetch(`${backendUrl}/trips`);
      if (r.ok) { const d = await r.json(); setTrips(Array.isArray(d) ? d : []); }
    } catch (e) {}
  };
  const fetchItinerary = async () => {
    try {
      const r = await fetch(`${backendUrl}/itinerary`);
      if (r.ok) { const d = await r.json(); setItinerary(Array.isArray(d) ? d : []); }
    } catch (e) {}
  };
  useEffect(() => { fetchTrips(); fetchItinerary(); }, []);

  const addTrip = async (e) => {
    e.preventDefault();
    const payload = { ...tripForm, budget: tripForm.budget ? parseFloat(tripForm.budget) : null, start_date: tripForm.start_date || null, end_date: tripForm.end_date || null };
    const newTrip = { ...payload, id: Date.now() };
    setTrips([...trips, newTrip]);
    try {
      await fetch(`${backendUrl}/trips`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      fetchTrips();
    } catch (e) {}
    setTripForm({ title: '', destination: '', start_date: '', end_date: '', budget: '', notes: '' });
  };

  const deleteTrip = async (id) => {
    setTrips(trips.filter(t => t.id !== id));
    setItinerary(itinerary.filter(i => i.trip_id !== id));
    setSelectedTrip(selectedTrip === id ? null : selectedTrip);
    try {
      await fetch(`${backendUrl}/trips/${id}`, { method: 'DELETE' });
      fetchTrips();
      fetchItinerary();
    } catch (e) {}
  };

  const addItineraryItem = async (e) => {
    e.preventDefault();
    const tid = selectedTrip || (trips[0] && trips[0].id);
    if (!tid) return;
    const payload = { ...itemForm, trip_id: String(tid), day: parseInt(itemForm.day) || 1 };
    const newItem = { ...payload, id: Date.now() };
    setItinerary([...itinerary, newItem]);
    try {
      await fetch(`${backendUrl}/itinerary`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      fetchItinerary();
    } catch (e) {}
    setItemForm({ trip_id: tid, day: (parseInt(itemForm.day) || 1) + 1, title: '', description: '', time_slot: '' });
  };

  const deleteItinerary = async (id) => {
    setItinerary(itinerary.filter(i => i.id !== id));
    try {
      await fetch(`${backendUrl}/itinerary/${id}`, { method: 'DELETE' });
      fetchItinerary();
    } catch (e) {}
  };

  const tripItems = selectedTrip ? itinerary.filter(i => i.trip_id === selectedTrip) : itinerary;

  return (
    <div className="travel-app">
      <section className="section">
        <h2 className="section-title">Trips</h2>
        <div className="card">
          <form onSubmit={addTrip}>
            <div className="form-grid">
              <div className="input-group">
                <label className="input-label">Title</label>
                <input className="input" value={tripForm.title} onChange={e => setTripForm({ ...tripForm, title: e.target.value })} placeholder="Trip name" required />
              </div>
              <div className="input-group">
                <label className="input-label">Destination</label>
                <input className="input" value={tripForm.destination} onChange={e => setTripForm({ ...tripForm, destination: e.target.value })} placeholder="City or country" required />
              </div>
            </div>
            <div className="form-grid">
              <div className="input-group">
                <label className="input-label">Start Date</label>
                <input type="date" className="input" value={tripForm.start_date} onChange={e => setTripForm({ ...tripForm, start_date: e.target.value })} />
              </div>
              <div className="input-group">
                <label className="input-label">End Date</label>
                <input type="date" className="input" value={tripForm.end_date} onChange={e => setTripForm({ ...tripForm, end_date: e.target.value })} />
              </div>
            </div>
            <div className="form-grid">
              <div className="input-group">
                <label className="input-label">Budget ($)</label>
                <input type="number" className="input" value={tripForm.budget} onChange={e => setTripForm({ ...tripForm, budget: e.target.value })} placeholder="Optional" />
              </div>
            </div>
            <div className="input-group">
              <label className="input-label">Notes</label>
              <textarea className="input textarea" value={tripForm.notes} onChange={e => setTripForm({ ...tripForm, notes: e.target.value })} placeholder="Trip notes..." rows={2} />
            </div>
            <button className="btn btn-primary" type="submit">Add Trip</button>
          </form>
        </div>
        <div className="trip-cards">
          {trips.map(t => (
            <div key={t.id} className={`trip-card ${selectedTrip === t.id ? 'selected' : ''}`} onClick={() => setSelectedTrip(t.id)}>
              <h3 className="trip-card-title">{t.title}</h3>
              <p className="trip-card-dest">{t.destination}</p>
              <p className="trip-card-dates">{formatDate(t.start_date)} – {formatDate(t.end_date)}</p>
              {t.budget && <p className="trip-card-budget">${t.budget}</p>}
              <button className="btn btn-danger btn-sm" onClick={ev => { ev.stopPropagation(); deleteTrip(t.id); }}>Delete</button>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Itinerary</h2>
        <div className="card">
          <form onSubmit={addItineraryItem}>
            <div className="form-grid">
              <div className="input-group">
                <label className="input-label">Trip</label>
                <select className="input" value={selectedTrip || ''} onChange={e => setSelectedTrip(e.target.value || null)}>
                  {trips.map(t => <option key={t.id} value={t.id}>{t.title}</option>)}
                </select>
              </div>
              <div className="input-group">
                <label className="input-label">Day</label>
                <input type="number" className="input" value={itemForm.day} onChange={e => setItemForm({ ...itemForm, day: e.target.value })} min={1} />
              </div>
            </div>
            <div className="form-grid">
              <div className="input-group">
                <label className="input-label">Activity</label>
                <input className="input" value={itemForm.title} onChange={e => setItemForm({ ...itemForm, title: e.target.value })} placeholder="Activity title" required />
              </div>
              <div className="input-group">
                <label className="input-label">Time</label>
                <input className="input" value={itemForm.time_slot} onChange={e => setItemForm({ ...itemForm, time_slot: e.target.value })} placeholder="e.g. 9:00 AM" />
              </div>
            </div>
            <div className="input-group">
              <label className="input-label">Description</label>
              <input className="input" value={itemForm.description} onChange={e => setItemForm({ ...itemForm, description: e.target.value })} placeholder="Optional details" />
            </div>
            <button className="btn btn-primary" type="submit">Add Activity</button>
          </form>
        </div>
        <div className="itinerary-list">
          {tripItems.sort((a,b)=>a.day-b.day).map(i => (
            <div key={i.id} className="itinerary-item">
              <div>
                <span className="itinerary-day">Day {i.day}</span>
                <span className="itinerary-time">{i.time_slot}</span>
                <strong className="itinerary-title">{i.title}</strong>
                {i.description && <p className="itinerary-desc">{i.description}</p>}
              </div>
              <button className="btn btn-danger btn-sm" onClick={() => deleteItinerary(i.id)}>Delete</button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function App() {
  const getBackendUrl = () => {
    const env = (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || '').trim();
    if (env) return env;
    if (typeof window !== 'undefined') {
      if (window.__BACKEND_URL__) return window.__BACKEND_URL__;
      return `${window.location.protocol}//${window.location.hostname}:5001`;
    }
    return 'http://localhost:5001';
  };
  const backendUrl = getBackendUrl();
  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Travel Planner</h1>
        <p className="app-subtitle">Plan your trips and daily itineraries</p>
      </header>
      <TripManager backendUrl={backendUrl} />
    </div>
  );
}

export default App;
