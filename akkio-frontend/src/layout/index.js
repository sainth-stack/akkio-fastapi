import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import './style.css'
import { Outlet, Navigate, useLocation } from "react-router-dom";
import { isAuthenticated, hasAdminAccess, hasPermission } from "../utils/auth";
import { refreshCurrentUser, isNetworkError } from "../utils/api";
import Spinner from "react-bootstrap/Spinner";

export function AdminLayout() {
  const location = useLocation();
  const [ready, setReady] = useState(false);
  const [apiOffline, setApiOffline] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!isAuthenticated()) {
        setReady(true);
        return;
      }
      try {
        await refreshCurrentUser();
        if (!cancelled) setApiOffline(false);
      } catch (e) {
        if (isNetworkError(e)) {
          console.warn('API unreachable; using cached session', e.message || e);
          if (!cancelled) setApiOffline(true);
        } else {
          console.warn('Could not refresh user profile', e);
        }
      }
      if (!cancelled) setReady(true);
    })();
    return () => { cancelled = true; };
  }, []);

  if (!ready) {
    return (
      <div className="d-flex align-items-center justify-content-center vh-100 bg-light">
        <Spinner animation="border" role="status" />
      </div>
    );
  }

  if (!isAuthenticated()) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (location.pathname.startsWith('/administration') && !hasAdminAccess()) {
    return <Navigate to="/welcome" replace />;
  }

  if (location.pathname.startsWith('/multi-agent') && !hasPermission('reports')) {
    return <Navigate to="/welcome" replace />;
  }

  if (location.pathname.startsWith('/app-builder') && !hasPermission('reports')) {
    return <Navigate to="/welcome" replace />;
  }

  if (
    (location.pathname.startsWith('/agentic-builder') || location.pathname.startsWith('/fullstack-builder'))
    && !hasPermission('reports')
  ) {
    return <Navigate to="/welcome" replace />;
  }

  return (
    <div className="row p-0 m-0">
      <div className="col-lg-12 col-md-12 col-sm-12 col-xs-12 p-0 m-0 bg-light">
        {apiOffline ? (
          <div
            className="px-3 py-2 text-center small"
            style={{ background: '#fef3c7', color: '#92400e', borderBottom: '1px solid #fcd34d' }}
          >
            API server unreachable (port 8000). Shell UI works; login, builder, and previews need the backend running.
          </div>
        ) : null}
        <Navbar />
        <div className="d-flex justify-content-between">
          <div>
            <Sidebar />
          </div>
          <div className="w-100 main-content">
            <Outlet />
          </div>
        </div>
      </div>
    </div>
  );
}
