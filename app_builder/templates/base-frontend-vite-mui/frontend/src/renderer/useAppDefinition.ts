import { useMemo } from 'react';
import definitionJson from '../mock/data.json';

export interface AppDefinition {
  version: number;
  meta: { appName: string; description: string; domain: string };
  theme: Record<string, unknown>;
  navigation: Array<{ label: string; path: string; icon: string; pageId: string }>;
  pages: Array<{ id: string; title: string; subtitle?: string; blocks: Array<Record<string, unknown>> }>;
  data?: Record<string, unknown>;
  mock?: Record<string, unknown>;
}

export function useAppDefinition() {
  const definition = useMemo(() => definitionJson as AppDefinition, []);
  return { definition, loading: false, error: null as string | null };
}

export function dataBag(definition: AppDefinition): Record<string, unknown> {
  return definition.data || definition.mock || {};
}

export function getMockSlice(definition: AppDefinition, key: string): unknown {
  return dataBag(definition)[key];
}
