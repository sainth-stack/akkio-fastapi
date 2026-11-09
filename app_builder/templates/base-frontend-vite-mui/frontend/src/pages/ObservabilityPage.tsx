import { useCallback, useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import InsightsIcon from '@mui/icons-material/Insights';
import RefreshIcon from '@mui/icons-material/Refresh';
import { Button } from '@mui/material';
import { useAppDefinition } from '../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';

type ObservabilityEvent = {
  id: string;
  timestamp: string;
  source: string;
  model: string;
  prompt: string;
  output: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
};

type ObservabilitySummary = {
  call_count: number;
  total_tokens: number;
  total_cost_usd: number;
  by_model: Array<{ model: string; calls: number; total_tokens: number; cost_usd: number }>;
};

function formatUsd(n: number): string {
  if (n >= 0.01) return `$${n.toFixed(2)}`;
  if (n > 0) return `$${n.toFixed(4)}`;
  return '$0.00';
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function ObservabilityPage() {
  const { definition } = useAppDefinition();
  const apiBase = getAkkioApiBase();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<ObservabilityEvent[]>([]);
  const [summary, setSummary] = useState<ObservabilitySummary | null>(null);

  const refresh = useCallback(async () => {
    if (!apiBase) {
      setError('Open this app from the Akkio host (/app/...) to load LLM observability.');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const headers: Record<string, string> = {};
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${apiBase}/observability?limit=100`, { headers });
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Could not load observability data'));
      setEvents(Array.isArray(data.events) ? data.events : []);
      setSummary(data.summary ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load observability');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (!definition) return null;

  const hasCapability =
    !definition.meta?.capabilities?.length ||
    definition.meta.capabilities.includes('llm_observability');

  return (
    <Box sx={{ maxWidth: 1100, mx: 'auto', pb: 10 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <InsightsIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Observability
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Button size="small" startIcon={<RefreshIcon />} onClick={() => void refresh()} disabled={loading}>
          Refresh
        </Button>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        In-app chat and LLM API calls: prompts, responses, token counts, and estimated cost by model.
      </Typography>

      {!hasCapability ? (
        <Typography color="text.secondary">
          This screen is available when the app was built with an observability request in the prompt.
        </Typography>
      ) : null}

      {loading && !events.length ? (
        <Stack alignItems="center" sx={{ py: 6 }}>
          <CircularProgress size={32} />
        </Stack>
      ) : null}

      {error ? (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      ) : null}

      {summary ? (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  LLM calls
                </Typography>
                <Typography variant="h4" fontWeight={700}>
                  {summary.call_count}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  Total tokens
                </Typography>
                <Typography variant="h4" fontWeight={700}>
                  {summary.total_tokens.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  Est. cost (USD)
                </Typography>
                <Typography variant="h4" fontWeight={700}>
                  {formatUsd(summary.total_cost_usd)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      ) : null}

      {summary?.by_model?.length ? (
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 3 }}>
          {summary.by_model.map((row) => (
            <Chip
              key={row.model}
              label={`${row.model}: ${row.total_tokens.toLocaleString()} tok · ${formatUsd(row.cost_usd)}`}
              size="small"
              variant="outlined"
            />
          ))}
        </Stack>
      ) : null}

      {!loading && !events.length && !error ? (
        <Card variant="outlined">
          <CardContent>
            <Typography color="text.secondary">
              No LLM calls logged yet. Use the in-app chat assistant; each response will appear here with
              tokens and estimated cost.
            </Typography>
          </CardContent>
        </Card>
      ) : null}

      {events.length > 0 ? (
        <TableContainer component={Card} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Source</TableCell>
                <TableCell>Model</TableCell>
                <TableCell align="right">In</TableCell>
                <TableCell align="right">Out</TableCell>
                <TableCell align="right">Cost</TableCell>
                <TableCell>Prompt</TableCell>
                <TableCell>Output</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {events.map((ev) => (
                <TableRow key={ev.id} hover>
                  <TableCell sx={{ whiteSpace: 'nowrap', fontSize: 12 }}>
                    {formatTime(ev.timestamp)}
                  </TableCell>
                  <TableCell>
                    <Chip label={ev.source} size="small" />
                  </TableCell>
                  <TableCell sx={{ fontSize: 12 }}>{ev.model}</TableCell>
                  <TableCell align="right">{ev.input_tokens}</TableCell>
                  <TableCell align="right">{ev.output_tokens}</TableCell>
                  <TableCell align="right">{formatUsd(ev.cost_usd)}</TableCell>
                  <TableCell sx={{ maxWidth: 220 }}>
                    <Typography variant="body2" noWrap title={ev.prompt}>
                      {ev.prompt || '—'}
                    </Typography>
                  </TableCell>
                  <TableCell sx={{ maxWidth: 280 }}>
                    <Typography variant="body2" noWrap title={ev.output}>
                      {ev.output || '—'}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : null}
    </Box>
  );
}
