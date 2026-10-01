import { Routes, Route } from 'react-router-dom';
import AppShell from './layout/AppShell';
import DashboardPage from './pages/DashboardPage';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<DashboardPage />} />
      </Route>
    </Routes>
  );
}
