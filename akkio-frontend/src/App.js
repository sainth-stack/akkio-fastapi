import './App.css';
import { AdminLayout } from './layout';
import { Login } from './pages/Auth/login';
import { BrowserRouter, Route, Routes, Navigate, useParams } from "react-router-dom";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { ToastContainer } from 'react-toastify';
import { UsageProvider } from './contexts/UsageContext';
import AppBuilder from './app_creator/AppBuilder';
import AppBuilderList from './app_creator/AppBuilderList';
import MiddleContent from './pages/entryPage';
import MultiAgent from './pages/MultiAgent';
import Administration from './pages/Administration';

function RedirectAgenticEdit() {
  const { id } = useParams();
  return <Navigate to={`/agentic-builder/edit/${id}`} replace />;
}

function App() {
  const clientId = '573823221354-d175srri1ta9un581atkp7b9qenst32u.apps.googleusercontent.com';
  return (
    <BrowserRouter>
      <UsageProvider>
        <GoogleOAuthProvider clientId={clientId}>
          <ToastContainer />
          <Routes>
            <Route path="/" element={<Login />} />
            <Route path="/login" element={<Login />} />
            <Route element={<AdminLayout />}>
              <Route path="welcome" element={<MiddleContent />} />
              <Route path="multi-agent" element={<MultiAgent />} />
              <Route path="app-builder" element={<AppBuilderList />} />
              <Route path="app-builder/new" element={<AppBuilder />} />
              <Route path="app-builder/edit/:id" element={<AppBuilder />} />
              <Route path="agentic-builder" element={<AppBuilderList builderKind="fullstack" />} />
              <Route path="agentic-builder/new" element={<AppBuilder builderKind="fullstack" />} />
              <Route path="agentic-builder/edit/:id" element={<AppBuilder builderKind="fullstack" />} />
              <Route path="fullstack-builder" element={<Navigate to="/agentic-builder" replace />} />
              <Route path="fullstack-builder/new" element={<Navigate to="/agentic-builder/new" replace />} />
              <Route path="fullstack-builder/edit/:id" element={<RedirectAgenticEdit />} />
              <Route path="administration/*" element={<Administration />} />
            </Route>
            <Route path="*" element={<Navigate to="/welcome" replace />} />
          </Routes>
        </GoogleOAuthProvider>
      </UsageProvider>
    </BrowserRouter>
  );
}

export default App;
