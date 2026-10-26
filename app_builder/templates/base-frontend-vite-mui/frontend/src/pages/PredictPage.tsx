import { useCallback, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Grid,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import type { AppDefinition } from '../renderer/useAppDefinition';
import { buildAppDefinitionPayload } from '../utils/ragDownload';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';
type PredictResult = {
  predicted_downtime_hours_next_period: number;
  predicted_mtbf_hours: number;
  predicted_production_output: number;
  risk_level: string;
  confidence: number;
  drivers: string[];
  recommendation: string;
};

const FIELDS = [
  { key: 'mtbf_hours', label: 'MTBF (hours)' },
  { key: 'mttr_hours', label: 'MTTR (hours)' },
  { key: 'downtime_hours', label: 'Downtime (hours)' },
  { key: 'open_work_orders', label: 'Open work orders' },
  { key: 'machine_load_pct', label: 'Machine load (%)' },
  { key: 'production_output', label: 'Production output' },
] as const;

function seedFromDefinition(definition: AppDefinition): Record<string, string> {
  const bag = definition.data || definition.mock || {};
  let kpis: Record<string, unknown> = {};
  Object.entries(bag).forEach(([k, v]) => {
    if (k.startsWith('kpis_') && v && typeof v === 'object' && !Array.isArray(v)) {
      kpis = { ...kpis, ...(v as Record<string, unknown>) };
    }
  });
  const num = (keys: string[], fallback: string) => {
    for (const key of keys) {
      if (kpis[key] != null) return String(kpis[key]).replace(/[^\d.]/g, '') || fallback;
    }
    return fallback;
  };
  return {
    mtbf_hours: num(['mtbf_hours', 'mtbf'], '240'),
    mttr_hours: num(['mttr_hours', 'mttr'], '1.8'),
    downtime_hours: num(['downtime_hours', 'downtime'], '2.5'),
    open_work_orders: num(['open_work_orders', 'work_orders'], '48'),
    machine_load_pct: num(['machine_load', 'oee'], '78'),
    production_output: num(['production_output', 'output'], '850'),
  };
}

export default function PredictPage({ definition }: { definition: AppDefinition }) {
  const apiBase = getAkkioApiBase();
  const [values, setValues] = useState(() => seedFromDefinition(definition));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const riskColor = useMemo(() => {
    if (!result) return 'default';
    if (result.risk_level === 'High') return 'error';
    if (result.risk_level === 'Medium') return 'warning';
    return 'success';
  }, [result]);

  const runPredict = useCallback(async () => {
    if (!apiBase) {
      setError('Open this app from the Akkio host to run predictions.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      const token = getAkkioAccessToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const features: Record<string, number> = {};
      FIELDS.forEach(({ key }) => {
        features[key] = Number(values[key]) || 0;
      });
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          features,
          app_definition: buildAppDefinitionPayload(definition),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(String(data.detail || 'Prediction failed'));
      setResult(data as PredictResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, [apiBase, definition, values]);

  return (
    <Box sx={{ maxWidth: 960, mx: 'auto', pb: 10 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <TrendingUpIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>Predict</Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Run synthetic forecasts on operational inputs. Use the chat agent for natural-language predictions
        or adjust values below and click Run prediction.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>Inputs</Typography>
              <Stack spacing={2}>
                {FIELDS.map(({ key, label }) => (
                  <TextField
                    key={key}
                    label={label}
                    type="number"
                    size="small"
                    value={values[key]}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                  />
                ))}
                <Button variant="contained" onClick={runPredict} disabled={loading}>
                  {loading ? 'Running…' : 'Run prediction'}
                </Button>
                {error ? <Typography color="error" variant="body2">{error}</Typography> : null}
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>Forecast</Typography>
              {!result ? (
                <Typography variant="body2" color="text.secondary">
                  Results appear here after you run a prediction or ask the chat agent to forecast.
                </Typography>
              ) : (
                <Stack spacing={1.5}>
                  <Chip label={`Risk: ${result.risk_level}`} color={riskColor as 'error' | 'warning' | 'success'} />
                  <Typography variant="body2">
                    Predicted downtime (next period): <strong>{result.predicted_downtime_hours_next_period} h</strong>
                  </Typography>
                  <Typography variant="body2">
                    Predicted MTBF: <strong>{result.predicted_mtbf_hours} h</strong>
                  </Typography>
                  <Typography variant="body2">
                    Predicted output: <strong>{result.predicted_production_output}</strong>
                  </Typography>
                  <Typography variant="body2">
                    Confidence: <strong>{(result.confidence * 100).toFixed(0)}%</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Drivers: {result.drivers.join(', ')}
                  </Typography>
                  <Typography variant="body2">{result.recommendation}</Typography>
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
