import { MenuItem, Paper, Stack, TextField } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import type { AppDefinition } from '../../renderer/useAppDefinition';
import { getDataSlice } from '../../renderer/useAppDefinition';
import { usePageFilters } from '../../renderer/PageFilterContext';

export type FilterDef = {
  id: string;
  label: string;
  filterType?: 'search' | 'dateRange' | 'select';
  optionsKey?: string;
  field?: string;
};

export default function FilterBar({
  filters,
  definition,
}: {
  filters: FilterDef[];
  definition: AppDefinition;
}) {
  const { filters: state, setFilters } = usePageFilters();

  return (
    <Paper sx={{ p: 2, mb: 3, borderRadius: 3, bgcolor: 'background.paper' }}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} flexWrap="wrap" useFlexGap>
        {filters.map((f) => {
          const t = f.filterType || 'search';
          const id = f.id.toLowerCase();
          if (t === 'select' && f.optionsKey) {
            const opts = (getDataSlice(definition, f.optionsKey) as Array<{ value: string; label: string }>) || [];
            const value = state.values[f.field || id] || '';
            return (
              <TextField
                key={f.id}
                select
                size="small"
                label={f.label}
                value={value}
                onChange={(e) => {
                  const v = e.target.value;
                  setFilters((prev) => ({
                    ...prev,
                    values: { ...prev.values, [f.field || id]: v },
                  }));
                }}
                sx={{ minWidth: 180 }}
              >
                <MenuItem value="">All</MenuItem>
                {opts.map((o) => (
                  <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                ))}
              </TextField>
            );
          }
          if (t === 'dateRange') {
            return (
              <Stack key={f.id} direction="row" spacing={1}>
                <TextField
                  size="small"
                  type="date"
                  label={`${f.label} from`}
                  InputLabelProps={{ shrink: true }}
                  value={state.values[`${f.field || id}:from`] || ''}
                  onChange={(e) => setFilters((p) => ({ ...p, values: { ...p.values, [`${f.field || id}:from`]: e.target.value } }))}
                />
                <TextField
                  size="small"
                  type="date"
                  label="To"
                  InputLabelProps={{ shrink: true }}
                  value={state.values[`${f.field || id}:to`] || ''}
                  onChange={(e) => setFilters((p) => ({ ...p, values: { ...p.values, [`${f.field || id}:to`]: e.target.value } }))}
                />
              </Stack>
            );
          }
          return (
            <TextField
              key={f.id}
              size="small"
              label={f.label}
              placeholder="Search…"
              value={state.search}
              onChange={(e) => setFilters((p) => ({ ...p, search: e.target.value }))}
              InputProps={{ startAdornment: <SearchIcon fontSize="small" sx={{ mr: 1, opacity: 0.5 }} /> }}
              sx={{ minWidth: 220, flex: 1 }}
            />
          );
        })}
      </Stack>
    </Paper>
  );
}
