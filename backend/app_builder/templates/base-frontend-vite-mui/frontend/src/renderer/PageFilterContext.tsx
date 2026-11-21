import React, { createContext, useContext, useMemo, useState } from 'react';

export type PageFilters = {
  search: string;
  line: string;
  status: string;
  dateFrom: string;
  dateTo: string;
};

const defaultFilters: PageFilters = {
  search: '',
  line: '',
  status: '',
  dateFrom: '',
  dateTo: '',
};

const PageFilterContext = createContext<{
  filters: PageFilters;
  setFilters: React.Dispatch<React.SetStateAction<PageFilters>>;
} | null>(null);

export function PageFilterProvider({ children }: { children: React.ReactNode }) {
  const [filters, setFilters] = useState<PageFilters>(defaultFilters);
  const value = useMemo(() => ({ filters, setFilters }), [filters]);
  return <PageFilterContext.Provider value={value}>{children}</PageFilterContext.Provider>;
}

export function usePageFilters() {
  const ctx = useContext(PageFilterContext);
  if (!ctx) {
    return { filters: defaultFilters, setFilters: () => {} };
  }
  return ctx;
}
