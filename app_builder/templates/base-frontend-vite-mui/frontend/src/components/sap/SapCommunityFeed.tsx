import { useMemo, useState } from 'react';
import {
  alpha,
  Box,
  Chip,
  Divider,
  Grid,
  InputAdornment,
  Link,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import type { AppDefinition } from '../../renderer/useAppDefinition';
import { getDataSlice } from '../../renderer/useAppDefinition';

type TicketRow = Record<string, unknown>;

function matchQuery(row: TicketRow, q: string): boolean {
  if (!q.trim()) return true;
  const low = q.toLowerCase();
  const blob = [
    row.title,
    row.description,
    row.snippet,
    row.ticket_id,
    row.component,
    row.category,
    row.product,
  ]
    .map((v) => String(v || '').toLowerCase())
    .join(' ');
  return low.split(/\s+/).every((t) => blob.includes(t));
}

function scoreQuery(row: TicketRow, q: string): number {
  if (!q.trim()) return 0;
  const low = q.toLowerCase();
  let s = 0;
  const title = String(row.title || '').toLowerCase();
  const desc = String(row.description || '').toLowerCase();
  for (const tok of low.split(/\s+/)) {
    if (title.includes(tok)) s += 3;
    if (desc.includes(tok)) s += 1;
  }
  return s;
}

export default function SapCommunityFeed({
  definition,
  dataKey,
  defaultQuery = '',
  title,
}: {
  definition: AppDefinition;
  dataKey: string;
  defaultQuery?: string;
  title?: string;
}) {
  const [query, setQuery] = useState(defaultQuery);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const allRows = useMemo(
    () => (getDataSlice(definition, dataKey) as TicketRow[]) || [],
    [definition, dataKey],
  );

  const filtered = useMemo(() => {
    const q = query.trim();
    const rows = allRows.filter((r) => matchQuery(r, q));
    if (q) {
      return [...rows].sort((a, b) => scoreQuery(b, q) - scoreQuery(a, q));
    }
    return rows;
  }, [allRows, query]);

  const selected = useMemo(() => {
    if (selectedId) {
      return allRows.find((r) => String(r.ticket_id) === selectedId) ?? filtered[0];
    }
    return filtered[0];
  }, [allRows, filtered, selectedId]);

  const suggested = String(selected?.suggested_answer || '').trim();
  const resultLabel = query.trim()
    ? `${filtered.length}+ results for '${query.trim()}'`
    : `${allRows.length} community questions`;

  return (
    <Box sx={{ mb: 4 }}>
      {title ? (
        <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1.5 }}>
          {title}
        </Typography>
      ) : null}

      <TextField
        fullWidth
        size="small"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search community tickets…"
        sx={{ mb: 2, maxWidth: 640 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" color="action" />
            </InputAdornment>
          ),
        }}
      />
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
        {resultLabel} · Tab: Community
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={7}>
          <Stack spacing={0} divider={<Divider />}>
            {filtered.map((row) => {
              const id = String(row.ticket_id);
              const active = selected && String(selected.ticket_id) === id;
              return (
                <Box
                  key={id}
                  onClick={() => setSelectedId(id)}
                  sx={(theme) => ({
                    py: 2,
                    px: 1,
                    cursor: 'pointer',
                    borderRadius: 1,
                    bgcolor: active ? alpha(theme.palette.primary.main, 0.06) : 'transparent',
                    '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.04) },
                  })}
                >
                  <Typography
                    variant="overline"
                    color="text.secondary"
                    sx={{ letterSpacing: 1, fontSize: '0.65rem' }}
                  >
                    SAP COMMUNITY
                  </Typography>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      color: 'primary.main',
                      fontWeight: 600,
                      lineHeight: 1.35,
                      mt: 0.5,
                    }}
                  >
                    {String(row.title)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {String(row.posted_at || row.created_at)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    {String(row.snippet || row.description)}
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 1.5 }}>
                    <Chip
                      size="small"
                      label={String(row.document_type || 'Question')}
                      variant="outlined"
                    />
                    {row.product ? (
                      <Chip size="small" label={String(row.product)} variant="outlined" />
                    ) : null}
                  </Stack>
                </Box>
              );
            })}
            {!filtered.length ? (
              <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                No tickets match this search.
              </Typography>
            ) : null}
          </Stack>
        </Grid>

        <Grid item xs={12} md={5}>
          <Paper
            elevation={0}
            sx={(theme) => ({
              p: 2,
              borderRadius: 2,
              bgcolor: alpha(theme.palette.primary.main, 0.08),
              border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
              position: { md: 'sticky' },
              top: 16,
            })}
          >
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Typography variant="subtitle2" fontWeight={700}>
                Suggested answer
              </Typography>
              <Chip label="Beta" size="small" color="secondary" sx={{ height: 22 }} />
            </Stack>
            <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 1.5 }}>
              <AutoAwesomeIcon sx={{ fontSize: 16, color: 'primary.main' }} />
              <Typography variant="caption" color="text.secondary">
                AI-generated
              </Typography>
            </Stack>
            {selected ? (
              <>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {suggested ||
                    'Select a community question or search to load the grounded suggested answer.'}
                </Typography>
                {selected.community_url ? (
                  <Link
                    href={String(selected.community_url)}
                    target="_blank"
                    rel="noopener noreferrer"
                    variant="caption"
                    sx={{ display: 'inline-block', mt: 2 }}
                  >
                    View on SAP Community
                  </Link>
                ) : null}
              </>
            ) : null}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}