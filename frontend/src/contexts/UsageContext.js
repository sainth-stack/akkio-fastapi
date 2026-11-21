import React, { createContext, useContext, useEffect, useState } from 'react';
import api, { getAccessToken } from '../utils/api';

const UsageContext = createContext(null);

export function UsageProvider({ children }) {
  const [usage, setUsage] = useState({
    credits_remaining: 0,
    storage_remaining_mb: 0,
    loading: true,
  });

  const loadUsage = async () => {
    if (!getAccessToken()) {
      setUsage((prev) => ({ ...prev, loading: false }));
      return;
    }
    try {
      const resp = await api.get('/usage', { skipAuthRedirect: true });
      const data = resp.data;
      if (data) {
        setUsage({
          credits_remaining: typeof data.credits_remaining === 'number' ? data.credits_remaining : 0,
          storage_remaining_mb: typeof data.storage_remaining_mb === 'number' ? data.storage_remaining_mb : 0,
          loading: false,
        });
      }
    } catch (e) {
      if (e.response?.status !== 401) {
        console.warn('Usage unavailable (API may be down):', e.message || e);
      }
      setUsage((prev) => ({ ...prev, loading: false }));
    }
  };

  useEffect(() => {
    loadUsage();

    const onUsageUpdate = () => loadUsage();
    window.addEventListener('usage_updated', onUsageUpdate);
    window.addEventListener('focus', onUsageUpdate);

    return () => {
      window.removeEventListener('usage_updated', onUsageUpdate);
      window.removeEventListener('focus', onUsageUpdate);
    };
  }, []);

  return (
    <UsageContext.Provider value={{ ...usage, refreshUsage: loadUsage }}>
      {children}
    </UsageContext.Provider>
  );
}

export function useUsage() {
  const context = useContext(UsageContext);
  if (!context) {
    throw new Error('useUsage must be used within a UsageProvider');
  }
  return context;
}
