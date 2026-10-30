import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import definitionJson from '../mock/data.json';
import { getAkkioAccessToken, getAkkioApiBase } from './akkioApi';

export interface AppDefinition {
  version: number;
  meta: {
    appName: string;
    description: string;
    domain: string;
    capabilities?: string[];
    dataSource?: string;
  };
  theme: Record<string, unknown>;
  navigation: Array<{ label: string; path: string; icon: string; pageId: string }>;
  pages: Array<{ id: string; title: string; subtitle?: string; blocks: Array<Record<string, unknown>> }>;
  data?: Record<string, unknown>;
  mock?: Record<string, unknown>;
}

export type ImportTarget = {
  dataKey: string;
  blockType: string;
  pageId: string;
  label: string;
};

type AppDefinitionContextValue = {
  definition: AppDefinition | null;
  loading: boolean;
  error: string | null;
  refreshDefinition: () => Promise<void>;
  importTargets: ImportTarget[];
};

const AppDefinitionContext = createContext<AppDefinitionContextValue | null>(null);

function collectImportTargets(defn: AppDefinition): ImportTarget[] {
  const targets: ImportTarget[] = [];
  const seen = new Set<string>();
  for (const page of defn.pages || []) {
    for (const block of page.blocks || []) {
      const dk = String(block.dataKey || '');
      if (!dk || seen.has(dk)) continue;
      seen.add(dk);
      targets.push({
        dataKey: dk,
        blockType: String(block.type || ''),
        pageId: page.id,
        label: String(block.title || block.type || dk),
      });
    }
  }
  return targets;
}

export function AppDefinitionProvider({ children }: { children: ReactNode }) {
  const staticDef = definitionJson as AppDefinition;
  const [definition, setDefinition] = useState<AppDefinition>(staticDef);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshDefinition = useCallback(async () => {
    const base = getAkkioApiBase();
    if (!base) {
      setDefinition(staticDef);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const token = getAkkioAccessToken();
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${base}/definition`, { headers });
      if (!res.ok) {
        if (res.status === 404) {
          setDefinition(staticDef);
          return;
        }
        throw new Error(`Could not load app data (${res.status})`);
      }
      const data = (await res.json()) as AppDefinition;
      setDefinition(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load definition');
      setDefinition(staticDef);
    } finally {
      setLoading(false);
    }
  }, [staticDef]);

  useEffect(() => {
    void refreshDefinition();
  }, [refreshDefinition]);

  const importTargets = useMemo(
    () => (definition ? collectImportTargets(definition) : []),
    [definition],
  );

  const value = useMemo(
    () => ({
      definition,
      loading,
      error,
      refreshDefinition,
      importTargets,
    }),
    [definition, loading, error, refreshDefinition, importTargets],
  );

  return <AppDefinitionContext.Provider value={value}>{children}</AppDefinitionContext.Provider>;
}

export function useAppDefinition(): AppDefinitionContextValue {
  const ctx = useContext(AppDefinitionContext);
  if (!ctx) {
    const definition = definitionJson as AppDefinition;
    return {
      definition,
      loading: false,
      error: null,
      refreshDefinition: async () => {},
      importTargets: collectImportTargets(definition),
    };
  }
  return ctx;
}

export function dataBag(definition: AppDefinition): Record<string, unknown> {
  return definition.data || definition.mock || {};
}

export function getDataSlice(definition: AppDefinition, key: string): unknown {
  return dataBag(definition)[key];
}

/** @deprecated use getDataSlice */
export function getMockSlice(definition: AppDefinition, key: string): unknown {
  return getDataSlice(definition, key);
}
