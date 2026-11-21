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
    dataModel?: string;
    generationPolicy?: string;
    containsSyntheticData?: boolean;
    /** Absent when Predict is not configured. JSON serialization may use null. */
    predictConfig?: Record<string, unknown> | null;
    sourceDatasets?: Array<{
      id: string;
      dataKey: string;
      title: string;
      description?: string | null;
      columns: Array<Record<string, unknown>>;
    }> | null;
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
  /** Replace UI state from server (e.g. after data upload). Does not call /definition. */
  applyDefinition: (next: AppDefinition) => void;
  /** Optional: fetch latest from GET /definition (off by default on app load). */
  refreshDefinition: () => Promise<void>;
  importTargets: ImportTarget[];
};

const AppDefinitionContext = createContext<AppDefinitionContextValue | null>(null);

function textValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' && value.trim() ? value : fallback;
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
}

/** Accept generated data.json, including null optional fields, and return a renderer-safe definition. */
export function readAppDefinition(raw: unknown): AppDefinition {
  const root = recordValue(raw) || {};
  const metaIn = recordValue(root.meta) || {};
  const pages = (Array.isArray(root.pages) ? root.pages : []).flatMap((item) => {
    const page = recordValue(item);
    if (!page) return [];
    const id = textValue(page.id, 'page');
    return [{
      id,
      title: textValue(page.title, id),
      subtitle: typeof page.subtitle === 'string' ? page.subtitle : undefined,
      blocks: Array.isArray(page.blocks) ? page.blocks.filter((block) => block && typeof block === 'object' && !Array.isArray(block)) as Array<Record<string, unknown>> : [],
    }];
  });
  const navigation = (Array.isArray(root.navigation) ? root.navigation : []).flatMap((item, index) => {
    const nav = recordValue(item);
    if (!nav) return [];
    const pageId = textValue(nav.pageId, pages[index]?.id || `page_${index + 1}`);
    return [{
      label: textValue(nav.label, pageId),
      path: index === 0 ? '/' : textValue(nav.path, `/${pageId}`),
      icon: textValue(nav.icon, 'Article'),
      pageId,
    }];
  });
  const theme: Record<string, unknown> = {};
  const themeIn = recordValue(root.theme);
  if (themeIn) {
    for (const [key, value] of Object.entries(themeIn)) {
      if (typeof value === 'string') theme[key] = value;
      if (key === 'chartPalette' && Array.isArray(value)) {
        theme[key] = value.filter((color): color is string => typeof color === 'string');
      }
    }
  }
  const sources = Array.isArray(metaIn.sourceDatasets)
    ? metaIn.sourceDatasets.flatMap((item) => {
        const ds = recordValue(item);
        if (!ds) return [];
        const dataKey = textValue(ds.dataKey);
        if (!dataKey) return [];
        return [{
          id: textValue(ds.id, dataKey),
          dataKey,
          title: textValue(ds.title, dataKey),
          description: typeof ds.description === 'string' ? ds.description : undefined,
          columns: Array.isArray(ds.columns)
            ? ds.columns.filter((col) => col && typeof col === 'object') as Array<Record<string, unknown>>
            : [],
        }];
      })
    : undefined;

  return {
    version: typeof root.version === 'number' ? root.version : 1,
    meta: {
      appName: textValue(metaIn.appName, 'Application'),
      description: typeof metaIn.description === 'string' ? metaIn.description : '',
      domain: textValue(metaIn.domain, 'generic'),
      capabilities: Array.isArray(metaIn.capabilities) ? metaIn.capabilities.filter((item): item is string => typeof item === 'string') : undefined,
      dataSource: typeof metaIn.dataSource === 'string' ? metaIn.dataSource : undefined,
      dataModel: typeof metaIn.dataModel === 'string' ? metaIn.dataModel : undefined,
      generationPolicy: typeof metaIn.generationPolicy === 'string' ? metaIn.generationPolicy : undefined,
      containsSyntheticData: metaIn.containsSyntheticData === true,
      predictConfig: recordValue(metaIn.predictConfig),
      sourceDatasets: sources,
    },
    theme,
    navigation: navigation.length ? navigation : [{ label: pages[0]?.title || 'Dashboard', path: '/', icon: 'Dashboard', pageId: pages[0]?.id || 'dashboard' }],
    pages: pages.length ? pages : [{ id: 'dashboard', title: 'Dashboard', blocks: [] }],
    data: recordValue(root.data),
    mock: recordValue(root.mock),
  };
}

function collectImportTargets(defn: AppDefinition): ImportTarget[] {
  const fromMeta = defn.meta?.sourceDatasets;
  if (Array.isArray(fromMeta) && fromMeta.length) {
    return fromMeta.map((ds) => ({
      dataKey: ds.dataKey,
      blockType: 'sourceTable',
      pageId: '',
      label: ds.title || ds.id || ds.dataKey,
    }));
  }
  const targets: ImportTarget[] = [];
  const seen = new Set<string>();
  for (const page of defn.pages || []) {
    for (const block of page.blocks || []) {
      const dk = String(block.dataKey || '');
      if (!dk || seen.has(dk) || dk.startsWith('source_')) continue;
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
  const staticDef = readAppDefinition(definitionJson);
  const [definition, setDefinition] = useState<AppDefinition>(staticDef);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyDefinition = useCallback((next: AppDefinition) => {
    setDefinition(next);
    setError(null);
  }, []);

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
        throw new Error(`Could not load app data (${res.status})`);
      }
      setDefinition(readAppDefinition(await res.json()));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load definition');
    } finally {
      setLoading(false);
    }
  }, [staticDef]);

  useEffect(() => {
    if (getAkkioApiBase()) void refreshDefinition();
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
      applyDefinition,
      refreshDefinition,
      importTargets,
    }),
    [definition, loading, error, applyDefinition, refreshDefinition, importTargets],
  );

  return <AppDefinitionContext.Provider value={value}>{children}</AppDefinitionContext.Provider>;
}

export function useAppDefinition(): AppDefinitionContextValue {
  const ctx = useContext(AppDefinitionContext);
  if (!ctx) {
    const definition = readAppDefinition(definitionJson);
    return {
      definition,
      loading: false,
      error: null,
      applyDefinition: () => {},
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
