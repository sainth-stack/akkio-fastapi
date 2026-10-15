import { tokens } from './tokens';
import { useAppDefinition } from '../renderer/useAppDefinition';

export function useChartPalette(): string[] {
  const { definition } = useAppDefinition();
  const fromDef = definition?.theme?.chartPalette;
  if (Array.isArray(fromDef) && fromDef.length) {
    return fromDef.map(String);
  }
  return [...tokens.chartPalette];
}
