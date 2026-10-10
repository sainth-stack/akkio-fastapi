/**
 * App.tsx — OVERWRITTEN by the deterministic generator.
 * This default version renders DemoPage so the template builds stand-alone.
 */
import { Routes, Route } from 'react-router-dom';
import AppShell from './components/ui/AppShell';
import DemoPage from './pages/DemoPage';

const NAV_ITEMS = [
  { label: 'Demo', path: '/', icon: 'Dashboard' },
];

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell navItems={NAV_ITEMS} appName="Agentic App" />}>
        <Route path="/" element={<DemoPage />} />
        <Route path="*" element={<DemoPage />} />
      </Route>
    </Routes>
  );
}
