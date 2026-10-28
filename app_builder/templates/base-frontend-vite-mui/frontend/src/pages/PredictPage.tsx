import { useCallback, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import SyncIcon from '@mui/icons-material/Sync';
import type { AppDefinition } from '../renderer/useAppDefinition';
import { buildAppDefinitionPayload } from '../utils/ragDownload';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';
import PredictionIndicators, { type LivePrediction } from '../components/predict/PredictionIndicators';

const FIELDS = [
  { key: 'mtbf_hours', label: 'MTBF (hours)' },
  { key: 'mttr_hours', label: 'MTTR (hours)' },
  { key: 'downtime_hours', label: 'Downtime (hours)' },
  { key: 'open_work_orders', label: 'Open work orders' },
  { key: 'machine_load_pct', label: 'Machine load / OEE (%)' },
  { key: 'production_output', label: 'Production output' },
] as const;

function baselineToForm(baseline: Record<string, number>): Record<string, string> {
  const out: Record<string, string> = {};
  FIELDS.forEach(({ key }) => {
    out[key] = String(baseline[key] ?? '');
  });
  return out;
}

export default function PredictPage({ definition }: { definition: AppDefinition }) {
  const apiBase = getAkkioApiBase();
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(FIELDS.map((f) => [f.key, ''])),
  );
  const [scenario, setScenario] = useState('');
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [result, setResult] = useState<LivePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [baselineLoaded, setBaselineLoaded] = useState(false);

  const postPredict = useCallback(
    async (body: Record<string, unknown>) => {
      if (!apiBase) throw new Error('Open this app from the Akkio host to run predictions.');
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          app_definition: buildAppDefinitionPayload(definition),
          ...body,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Prediction failed'));
      return data as LivePrediction & { baseline?: Record<string, number> };
    },
    [apiBase, definition],
  );

  const syncFromLiveData = useCallback(async () => {
    setSyncing(true);
    setError(null);
    try {
      const data = await postPredict({ refresh_baseline_only: true });
      if (data.baseline) {
        setValues(baselineToForm(data.baseline));
        setBaselineLoaded(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load live KPIs');
    } finally {
      setSyncing(false);
    }
  }, [postPredict]);

  const runPredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const features: Record<string, number> = {};
      FIELDS.forEach(({ key }) => {
        const raw = values[key];
        if (raw !== '' && raw != null) {
          features[key] = Number(raw) || 0;
        }
      });
      const data = await postPredict({
        features,
        prompt: scenario.trim(),
      });
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, [postPredict, scenario, values]);

  return (
    <Box sx={{ maxWidth: 1000, mx: 'auto', pb: 10 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <TrendingUpIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>Predict</Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Forecasts stay in the same range as your dashboard KPIs. Use <strong>Sync live KPIs</strong>, then
        adjust inputs slightly or describe a small what-if (e.g. load +10%). Extreme values are clamped to
        a realistic band vs baseline.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Card variant="outlined">
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight={600}>Inputs</Typography>
                <Button
                  size="small"
                  startIcon={<SyncIcon />}
                  onClick={syncFromLiveData}
                  disabled={syncing}
                >
                  {syncing ? 'Syncing…' : 'Sync live KPIs'}
                </Button>
              </Stack>
              <Stack spacing={2}>
                <TextField
                  label="Scenario prompt (optional)"
                  placeholder="What if machine load is 92% and downtime doubles?"
                  size="small"
                  multiline
                  minRows={2}
                  value={scenario}
                  onChange={(e) => setScenario(e.target.value)}
                />
                {FIELDS.map(({ key, label }) => (
                  <TextField
                    key={key}
                    label={label}
                    type="number"
                    size="small"
                    placeholder={baselineLoaded ? undefined : 'Sync live KPIs or run predict'}
                    value={values[key]}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                  />
                ))}
                <Button variant="contained" onClick={runPredict} disabled={loading || !baselineLoaded}>
                  {loading ? 'Running forecast…' : 'Run prediction'}
                </Button>
                {!baselineLoaded ? (
                  <Typography variant="caption" color="text.secondary">
                    Sync live KPIs first so inputs and forecasts match your dashboard scale.
                  </Typography>
                ) : null}
                {error ? <Typography color="error" variant="body2">{error}</Typography> : null}
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={7}>
          <Card variant="outlined" sx={{ minHeight: 320 }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>Forecast indicators</Typography>
              {loading ? (
                <Typography variant="body2" color="text.secondary">Computing forecast from live data…</Typography>
              ) : !result ? (
                <Typography variant="body2" color="text.secondary">
                  No forecast yet. Sync live KPIs, add a scenario if you want, then run prediction.
                  Chat forecasts also recompute on each message (you will see &quot;Thinking…&quot;).
                </Typography>
              ) : (
                <PredictionIndicators result={result} />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
